# Copyright Swiss National Supercomputing Centre (CSCS/ETH Zurich)
# ReFrame Project Developers. See the top-level LICENSE file for details.
#
# SPDX-License-Identifier: BSD-3-Clause
import os

import reframe as rfm
import reframe.utility.sanity as sn
from uenv import uarch

gromacs_references = {
    'STMV': {
        'gh200': {
            'perf': (117, -0.02, None, 'ns/day'),
            'bondenergy': (164780, -0.01, None, 'kJ/mol'),
        }
    },
 }

slurm_config = {
    'STMV': {
        'gh200': {
            'nodes': 1,
            'ntasks-per-node': 8,
            'cpus-per-task': 32,
            'walltime': '0d0h5m0s',
            'gpu': True,
        }
    },
    'hEGFRDimerPair': {
        'gh200': {
            'nodes': 1,
            'ntasks-per-node': 8,
            'cpus-per-task': 32,
            'walltime': '0d0h5m0s',
            'gpu': True,
        }
    },
}


class gromacs_download(rfm.RunOnlyRegressionTest):
    descr = 'Fetch GROMACS source code'
    maintainers = ['SSA']
    version = variable(str, value='2024.3')
    sourcesdir = None
    executable = 'wget'
    executable_opts = [
        '--quiet',
        'https://jfrog.svc.cscs.ch/artifactory/cscs-reframe-tests/gromacs/'
        f'v{version}.tar.gz',
        # https://github.com/gromacs/gromacs/archive/refs/tags/v2024.3.tar.gz
    ]
    local = True

    @sanity_function
    def validate_download(self):
        return sn.assert_eq(self.job.exitcode, 0)


@rfm.simple_test
class gromacs_build_test(rfm.CompileOnlyRegressionTest):
    """
    Test GROMACS build from source using the develop view
    """
    descr = 'GROMACS Build Test'
    valid_prog_environs = ['+gromacs-dev']
    valid_systems = ['*']
    build_system = 'CMake'
    maintainers = ['SSA']
    sourcesdir = None
    gromacs_sources = fixture(gromacs_download, scope='session')
    build_locally = False
    tags = {'uenv', 'production'}

    @run_before('compile')
    def prepare_build(self):
        self.uarch = uarch(self.current_partition)
        self.skip_if_no_procinfo()
        cpu = self.current_partition.processor
        self.build_system.max_concurrency = cpu.info['num_cpus_per_socket']
        tarsource = os.path.join(
            self.gromacs_sources.stagedir,
            f'v{self.gromacs_sources.version}.tar.gz'
        )
        self.prebuild_cmds = [f'tar --strip-components=1 -xzf {tarsource}']
        self.build_system.config_opts = [
            '-DREGRESSIONTEST_DOWNLOAD=ON',
            '-DGMX_MPI=on',
            '-DGMX_BUILD_OWN_FFTW=ON',
            '-DCP2K_USE_SPGLIB=ON',
            '-DGMX_HWLOC=ON',
            '-DGMX_SIMD=ARM_NEON_ASIMD',
            '-DGMX_INSTALL_NBLIB_API=ON',
        ]
        if self.uarch == 'gh200':
            self.build_system.config_opts += ['-DGMX_GPU=CUDA']

    @sanity_function
    def validate_test(self):
        self.gromacs_executable = os.path.join('bin', 'gmx_mpi')
        return os.path.isfile(self.gromacs_executable)


@rfm.simple_test
class gromacs_run_test(rfm.RunOnlyRegressionTest):
    executable = './mps-wrapper.sh -- gmx_mpi mdrun -s topol.tpr'
    executable_opts = ['-dlb no', '-ntomp 32', '-pme gpu', '-npme 1',
                       '-bonded gpu', '-nb gpu', '-nsteps 10000', '-update gpu',
                       '-pin off', '-v', '-noconfout', '-nstlist 300']
    maintainers = ['SSA']
    valid_systems = ['*']
    test_name = variable(str, value='STMV')
    valid_prog_environs = ['+gromacs']
    tags = {'uenv', 'production'}

    @run_before('run')
    def prepare_run(self):
        self.uarch = uarch(self.current_partition)
        config = slurm_config[self.test_name][self.uarch]
        self.job.options = [
            f'--nodes={config["nodes"]}',
        ]
        self.num_tasks_per_node = config['ntasks-per-node']
        self.num_tasks = config['nodes'] * self.num_tasks_per_node
        self.num_cpus_per_task = config['cpus-per-task']
        self.time_limit = config['walltime']
        # environment variables
        self.env_vars['OMP_NUM_THREADS'] = str(self.num_cpus_per_task)
        self.env_vars['FI_CXI_RX_MATCH_MODE'] = 'software'
        if self.uarch == 'gh200':
            self.env_vars['MPICH_GPU_SUPPORT_ENABLED'] = '1'
            self.env_vars['GMX_GPU_DD_COMMS'] = 'true'
            self.env_vars['GMX_GPU_PME_PP_COMMS'] = 'true'
            self.env_vars['GMX_ENABLE_DIRECT_GPU_COMM'] = '1'
            self.env_vars['GMX_FORCE_GPU_AWARE_MPI'] = '1'

        # set reference
        if self.uarch is not None and \
           self.uarch in gromacs_references[self.test_name]:
            self.reference = {
                self.current_partition.fullname:
                    gromacs_references[self.test_name][self.uarch]
            }

        # set input files
        if self.test_name == 'STMV':
            jfrog = 'https://jfrog.svc.cscs.ch/artifactory/cscs-reframe-tests/'
            self.prerun_cmds = [f'wget {jfrog}/gromacs/STMV/topol.tpr --quiet']

    @run_after('setup')
    def postproc_run(self):
        """
        Extract Bond Energy from the binary .edr file and write it to a
        readable .xvg file and then write it to stdout
        """
        self.postrun_cmds = [
            'echo -e "1\\n" |'
            'gmx_mpi energy -f ener.edr -o ener.xvg',
            'cat ener.xvg'
        ]

    @sanity_function
    def validate_job(self):
        regex1 = r'^Bond\s+\S+\s+'
        regex2 = r'^Performance:\s+\S+\s+'
        self.sanity_patterns = sn.all([
            sn.assert_found(regex1, self.stdout, msg='regex1 failed'),
            sn.assert_found(regex2, self.stderr, msg='regex2 failed'),
        ])
        return self.sanity_patterns

    @performance_function('ns/day')
    def perf(self):
        regex = r'^Performance:\s+(?P<ns_day>\S+)\s+'
        return sn.extractsingle(regex, self.stderr, 'ns_day', float)

    @performance_function('kJ/mol')
    def bondenergy(self):
        regex = r'^Bond\s+(?P<bond_energy>\S+)\s+'
        return sn.extractsingle(regex, self.stdout, 'bond_energy', int)

@rfm.simple_test
class gromacs_stmv(rfm.RunOnlyRegressionTest):
    benchmark_info = {
        'name': 'stmv_v2',
        'energy_step0_ref': -1.46491e+07,
        'energy_step0_tol': 0.001,
    }

    valid_systems = ['*']
    valid_prog_environs = ['+gromacs']

    maintainers = ['SSA', 'RMeli']
    use_multithreading = False
    exclusive_access = True
    num_nodes = parameter([1], loggable=True)
    num_gpus_per_node = 4
    time_limit = '10m'
    nb_impl = parameter(['gpu'])
    update_mode = parameter(['gpu'])
    bonded_impl = 'gpu'
    executable = 'gmx_mpi mdrun'
    tags = {'uenv', 'production', 'adac'}
    keep_files = ['md.log']
    test_name = variable(str, value='STMV')

    allref = {
        1: {
            'gpu': (100.0, -0.05, None, 'ns/day'), # update=gpu, gpu resident mode
        },
        2: {
            'gpu': (76.9, -0.05, None, 'ns/day'), # update=gpu, gpu resident mode
        },
    }

    @run_after('init')
    def prepare_test(self):
        self.descr = f"GROMACS {self.benchmark_info['name']} benchmark (update mode: {self.update_mode}, bonded: {self.bonded_impl}, non-bonded: {self.nb_impl})"
        bench_file_path = os.path.join(self.current_system.resourcesdir, 
                                      'datasets',
                                      'gromacs',
                                       self.benchmark_info['name'],
                                      'pme_nvt.tpr')
        self.prerun_cmds = [
            f'ln -s {bench_file_path} benchmark.tpr'
        ]

    @run_after('init')
    def setup_runtime(self):
        self.num_tasks_per_node = 4
        self.num_tasks = self.num_tasks_per_node*self.num_nodes
        npme_ranks = 1

        self.executable_opts += [
            '-nsteps -1',
            '-maxh 0.085',  # sets runtime to 5 mins
            '-nstlist 400', # refer to https://manual.gromacs.org/2024.1/user-guide/mdp-options.html#mdp-nstlist
            '-noconfout',
            '-notunepme',
            '-resetstep 10000',
            '-nb', self.nb_impl,
            '-npme', f'{npme_ranks}',
            '-pme', 'gpu',
            '-update', self.update_mode,
            '-bonded', self.bonded_impl,
            '-s benchmark.tpr'
        ]
        self.env_vars = {
            'MPICH_GPU_SUPPORT_ENABLED': '1',
            'OMP_NUM_THREADS': '64',
            'OMP_PROC_BIND': 'close',
            'OMP_PLACES': 'cores',
            'GMX_ENABLE_DIRECT_GPU_COMM': '1',
            'GMX_FORCE_GPU_AWARE_MPI': '1',
            'GMX_GPU_PME_DECOMPOSITION': '1',
            'GMX_PMEONEDD': '1',
        }

    @run_before('run')
    def set_cpu_mask(self):
        self.job.launcher.options = [f'--cpu-bind=core']

    @run_after('init')
    def add_select_gpu_wrapper(self):
        self.prerun_cmds += [
            'cat << EOF > select_gpu',
            '#!/bin/bash',
            'export CUDA_VISIBLE_DEVICES=\$SLURM_LOCALID',
            'exec \$*',
            'EOF',
            'chmod +x ./select_gpu'
        ]
        self.executable = './select_gpu ' + self.executable

    @performance_function('ns/day')
    def perf(self):
        return sn.extractsingle(r'Performance:\s+(?P<perf>\S+)',
                                'md.log', 'perf', float)

    @deferrable
    def energy_step0(self):
        return sn.extractsingle(r'\s+Kinetic En\.\s+Total Energy\s+Conserved En\.\s+Temperature\s+Pressure \(bar\)\n'
                                r'(\s+\S+)\s+(?P<energy>\S+)(\s+\S+){3}\n'
                                r'\s+Constr\. rmsd',
                                'md.log', 'energy', float)

    @deferrable
    def energy_drift(self):
        return sn.extractsingle(r'\s+Conserved\s+energy\s+drift\:\s+(\S+)', 'md.log', 1, float)

    @deferrable
    def verlet_buff_tol(self):
        return sn.extractsingle(r'\s+verlet-buffer-tolerance\s+\=\s+(\S+)', 'md.log', 1, float)

    @sanity_function
    def assert_energy_readout(self):
        return sn.all([
            sn.assert_found('Finished mdrun', 'md.log', 'Run failed to complete'), 
            sn.assert_reference(
                self.energy_step0(),
                self.benchmark_info['energy_step0_ref'],
                -self.benchmark_info['energy_step0_tol'],
                self.benchmark_info['energy_step0_tol']
            ),
            sn.assert_lt(
                self.energy_drift(),
                2*self.verlet_buff_tol()
           ),
        ])

    @run_before('run')
    def setup_run(self):
        try:
            found = self.allref[self.num_nodes][self.update_mode]
        except KeyError:
            self.skip(f'Configuration with {self.num_nodes} node(s) of '
                      f'{self.bench_name!r} is not supported on {arch!r}')

        # Setup performance references
        self.reference = {
            '*': {
                'perf': self.allref[self.num_nodes][self.update_mode]
            }
        }
