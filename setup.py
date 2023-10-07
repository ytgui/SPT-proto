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
        pytest.main([
            '-nauto', '-x',
            '--cov=naive_gpt',
            '--tb=long', 'test/'
        ])


setup(
    name='naive_gpt',
    version='0.1.0',
    packages=['naive_gpt'],
    ext_modules=[
        cpp_extension.CUDAExtension(
            'naive_gpt.ext',
            sources=[
                'extension/entry.cpp',
                'extension/softmax.cu',
                'extension/cdist.cu',
                'extension/lookup.cu',
                'extension/sddmm.cpp',
                'extension/spmm.cpp'
            ]
        )
    ] if torch.cuda.is_available() else [],
    cmdclass={
        'test': TestCommand,
        'build_ext': cpp_extension.BuildExtension,
    },
    install_requires=[
    ]
)
