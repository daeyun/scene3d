import time
import subprocess
import os
import tempfile
import random
import string
import shlex
from os import path
from scene3d import io_utils
from scene3d import exec_utils
from scene3d import log
import uuid
import textwrap


def run_commands_over_ssh(hostname, bash_command_lines):
    """
    Run shell commands through ssh and return stdout.
    Output will depend on

    This is very hackish and not secure at all. Use at your own risk.
    For now, it assumes there is no password or passphrase required (set up keys through ~/.ssh/config).
    :param hostname: hostname, according to ~/.ssh/config
    :param bash_command_lines: List of strings.
    :return: stdout of ssh command.
    """

    # Timestamp in milliseconds and random string.
    # We mostly want to avoid collision. And also make it easier to debug based on timestamps.
    random_code = '{}_{}'.format(int(time.time() * 1000), uuid.uuid4().hex[:12])
    magic_line_start = 'START #-## {}'.format(random_code)
    magic_line_end = 'END #-## {}'.format(random_code)

    block_token = 'ENDSSH'

    command_string = '\n'.join(bash_command_lines)
    assert block_token not in command_string

    content = textwrap.dedent('''
    ssh -T {hostname} '/bin/bash --noprofile --norc' <<-'ENDSSH'
    echo "{magic_start}"
    set -ex
    {command_string}
    echo "{magic_end}"
    ENDSSH
    ''').format(hostname=hostname,
                command_string=command_string,
                magic_start=magic_line_start,
                magic_end=magic_line_end).strip()

    tmp_dir = '/tmp/gen_sh/'
    io_utils.ensure_dir_exists(tmp_dir)

    filename = '{}.sh'.format(random_code)
    writepath = path.join(tmp_dir, filename)
    log.info('Saving temporary script: {}'.format(writepath))

    try:
        with open(writepath, 'w') as f:
            f.write(content)

        p, stdout, stderr = exec_utils.run_command(
            ['bash', '--noprofile', '--norc', writepath]
        )

        try:
            assert magic_line_start in stdout
            assert magic_line_end in stdout
        except AssertionError:
            raise RuntimeError('`magic_line_end` not found in stdout')

        out_lines = []
        seen_start = False
        for line in stdout.split('\n'):
            if seen_start:
                if magic_line_end in line:
                    break
                else:
                    out_lines.append(line)
            elif magic_line_start in line:
                seen_start = True
    except Exception as ex:
        raise ex

    if path.isfile(writepath):
        os.remove(writepath)

    return '\n'.join(out_lines)


def list_files(hostname, dirname, extra_flags=''):
    """
    Raises RuntimeError if `dirname` does not exist on the server.

    :param hostname: hostname, according to ~/.ssh/config
    :param dirname: Will be escaped. Home directory expansion and variables won't work.
    :param extra_flags: e.g. "lh"
    :return: Lines returned by the ls command.
    """
    assert "'" not in dirname
    assert '-' not in extra_flags
    assert ' ' not in extra_flags
    out = run_commands_over_ssh(hostname, [
        "ls -1A{} {}".format(extra_flags, shlex.quote(dirname))
    ])

    lines = out.split('\n')
    return lines


def sync_files_remote_to_local(hostname, source_dirname, local_dirname, includes=None):
    """
    :param hostname: hostname, according to ~/.ssh/config
    :param source_dirname: full path
    :param local_dirname: full path
    :param includes: list of patterns or filenames to include
    :return:
    """
    assert source_dirname.startswith('/')
    assert local_dirname.startswith('/')

    source_dirname = source_dirname.rstrip('/')
    local_dirname = local_dirname.rstrip('/')

    # basic sanity check to avoid those cases. there's probably a better way to do this.
    assert ' ' not in source_dirname
    assert ' ' not in local_dirname
    assert '*' not in source_dirname
    assert '*' not in local_dirname

    io_utils.ensure_dir_exists(local_dirname)

    # Notice the trailing slash in the source.
    command = ['rsync', '-av', '{}:{}/'.format(hostname, source_dirname), local_dirname]

    tmp_dir = '/tmp/sync_remote_to_local/'
    io_utils.ensure_dir_exists(tmp_dir)
    random_code = '{}_{}'.format(int(time.time() * 1000), uuid.uuid4().hex[:12])
    tmp_filename = path.join(tmp_dir, '{}.txt'.format(random_code))  # only used when `includes` is not None.

    if not includes:
        includes = None

    if includes is not None:
        assert isinstance(includes, (list, tuple))
        assert len(includes) > 0
        assert isinstance(includes[0], str)
        with open(tmp_filename, 'w') as f:
            f.write('\n'.join(includes))
        command.insert(2, '--include-from={}'.format(tmp_filename))
        command.insert(3, '--exclude=*')  # No quotes around *. Those are for the shell, not rsync.

    p, stdout, stderr = exec_utils.run_command(
        command=command
    )
    log.info('rsync stdout:\n{}'.format(stdout))

    if path.isfile(tmp_filename):
        os.remove(tmp_filename)

    return stdout
