import subprocess
import time
import textwrap
from os import path

from scene3d import log


def run_command(command, cwd=None):
    """
    Run command and return output.

    :param command: A list of executable and arguments.
    :param cwd: Current working directory.
    :return: Popen object, content of stdout, content of stderr
    """
    assert isinstance(command, (tuple, list))
    if cwd is not None:
        log.debug('CWD: {}'.format(cwd))
    log.debug(subprocess.list2cmdline(command))

    start_time = time.time()
    p = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=cwd)
    return_code = p.wait()
    log.debug('return code: {}'.format(return_code))
    elapsed = time.time() - start_time

    stdout, stderr = p.communicate()
    stdout = stdout.decode('utf-8')
    stderr = stderr.decode('utf-8')

    if p.returncode != 0:
        exec_summary = textwrap.dedent(textwrap.dedent("""
        Command: {}
        Return code: {}

        stdout:
        {}

        stderr:
        {}
        """).format(
            subprocess.list2cmdline(command),
            p.returncode,
            stdout,
            stderr
        ))
        raise RuntimeError(exec_summary)

    log.info('{0} took {1:.3f} seconds.'.format(path.basename(command[0]), elapsed))

    return p, stdout, stderr
