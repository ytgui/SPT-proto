import torch
from setuptools import setup, Command
from torch.utils import cpp_extension


class TestCommand(Command):
    description = "test"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        import pytest
        import coverage

        #
        cov = coverage.Coverage(
            config_file=True
        )
        #
        cov.start()
        ret = pytest.main(['-x', 'test'])
        cov.stop()
        #
        if ret == pytest.ExitCode.OK:
            cov.report()


setup(
    name='naive_gpt',
    version='0.1.0',
    packages=['naive_gpt'],
    ext_modules=[
        cpp_extension.CUDAExtension(
            'naive_gpt.ext',
            sources=[
                'extension/entry.cpp',
                'extension/sparse_mha.cu',
            ]
        )
    ],
    cmdclass={
        'test': TestCommand,
        'build_ext': cpp_extension.BuildExtension,
    },
    install_requires=[
    ]
)
