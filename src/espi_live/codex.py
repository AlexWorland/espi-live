"""Open the Codex command-line tool on the program's source code.

``find_cli`` looks for the ``codex`` command and ``open_cli`` starts it in a
new terminal window with the source folder as its working directory (a
program started from the desktop does not always see the same ``PATH`` as a
terminal, so the usual install folders are searched too). Before a downloaded
source copy is opened, ``write_guide`` puts ``AGENTS.override.md`` in it.
Codex reads that file instead of ``AGENTS.md`` and learns that the person is
a user, not a developer: answer from the code in plain words, ask before
sending a change, and send it with this program's ``--send-changes``
(``changes.py``). A source checkout opens as it is, with the developers'
``AGENTS.md``. No GUI imports: the front end reports what happened.
"""

import os
import shlex
import shutil
import subprocess
import sys

from espi_live import source_code, updater

INSTALL_URL = "https://developers.openai.com/codex/cli"
CLI_NAME = "codex"
# Where the command usually lives when the program was not started from a shell.
EXTRA_BIN_DIRS = ("/opt/homebrew/bin", "/usr/local/bin", "~/.local/bin", "~/.npm-global/bin",
                  "~/.bun/bin", "~/.volta/bin")
# How each Linux terminal is told to run a command: its name and the arguments
# that come before the command.
LINUX_TERMINALS = (("x-terminal-emulator", ("-e",)), ("gnome-terminal", ("--",)),
                   ("konsole", ("-e",)), ("xfce4-terminal", ("-x",)), ("xterm", ("-e",)))
GUIDE = """\
# You are helping a user of ESPI live

<!-- Written by ESPI live's "Open in Codex" button, which rewrites it each time. -->

The person in this chat uses **ESPI live**, a program that shows live video
from a FLIR camera for ESPI (electronic speckle pattern interferometry)
measurements. They opened this folder from the program's **Open in Codex**
button because they are curious how it works, and they may want to suggest a
change. Assume they are not a programmer.

This folder is a copy of the source code of the program they run
({build}, commit `{commit}`). Editing it does not change their program:
changes reach it only after the developers review and release them.

## Answering questions

- Answer from the code in this folder, not from general knowledge: read the
  relevant files first. `README.md` describes the features, `AGENTS.md` maps
  the code in `src/espi_live/`, and `docs/camera/` covers the camera.
- Use plain language and short answers. Explain any technical term you need.
  Name the file a behavior comes from (for example `src/espi_live/engine.py`)
  so a developer can follow up, but don't paste long code unless asked.
- If the answer is not in this code (for example the camera's firmware), say
  so.

## Making a change

Change files only when the person asks for a change.

1. Read `AGENTS.md` first. It holds the developers' rules for this code;
   follow them, except: never use git or gh, never commit, push or open pull
   requests yourself, and skip its screenshot rule.
2. Keep the change small and limited to what they asked for. Update the tests
   and documents `AGENTS.md` asks for. Don't change anything under `.github/`:
   such changes cannot be sent.
3. If Python 3 with numpy, opencv-python and PySide6 is installed, run the
   checks `AGENTS.md` lists. If not, don't install anything; say that the
   developers will test the change.
4. Explain in plain words what you changed and what it will do, then ask:
   **"Are you happy with these changes? If you are, I'll send them to the
   ESPI live developers to review."**
5. If they want something different, adjust it and ask again. Don't send
   anything until they clearly say yes.

## Sending a change (only after a clear yes)

1. Write `{request}` in this folder: a short title on the first line (a
   plain sentence under 80 characters), a blank line, then what changed and
   why in plain words, and anything you could not test.
2. Run this command. It needs internet access; if the sandbox blocks it, ask
   to run it outside the sandbox and explain that it sends the change to the
   ESPI live developers on GitHub:

   ```
   {command}
   ```

   It compares this folder with the original download, sends the
   differences with your description to the developers, and prints a link.
   No GitHub account, git or token is needed.
3. Tell them it was sent, give them the link, and say that the developers
   will review it and that it reaches their program only in a later update.
4. If the command fails, explain the error simply. Don't try other ways of
   sending it (git, gh, curl or the GitHub website).

Files here that are not source code: `.espi/` (the original download, used to
find what changed: never edit it), this file and `{request}`.
"""


def find_cli(platform=None):
    """Find the ``codex`` command.

    Args: platform is a sys.platform value; None uses the running one.
    Returns: The command's absolute path, or None when it is not installed.
    """
    found = shutil.which(CLI_NAME)
    if found is None and (platform or sys.platform) != "win32":
        extra = os.pathsep.join(os.path.expanduser(d) for d in EXTRA_BIN_DIRS)
        found = shutil.which(CLI_NAME, path=extra)
    return found


def apple_script_quote(text):
    """Quote text as an AppleScript string literal.

    Args: text is the text to quote.
    Returns: The text in double quotes, with backslashes and quotes escaped.
    """
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def terminal_command(exe, folder, platform=None, which=shutil.which):
    """Build the command that runs Codex in a new terminal window.

    Args: exe is the codex command's path; folder is the working directory;
        platform is a sys.platform value (None uses the running one); which
        finds a terminal program on Linux.
    Returns: A (command line, subprocess options) tuple. Raises OSError when
        Linux has no terminal program to use.
    """
    platform = platform or sys.platform
    if platform == "win32":
        # /k keeps the window open if Codex stops at once, so its error can be read.
        return (["cmd.exe", "/k", exe],
                {"cwd": folder, "creationflags": getattr(subprocess, "CREATE_NEW_CONSOLE", 0x10)})
    if platform == "darwin":
        shell = f"cd {shlex.quote(folder)} && {shlex.quote(exe)}"
        return (["osascript", "-e",
                 f"tell application \"Terminal\" to do script {apple_script_quote(shell)}",
                 "-e", 'tell application "Terminal" to activate'], {})
    for name, lead in LINUX_TERMINALS:
        terminal = which(name)
        if terminal:
            return ([terminal, *lead, "sh", "-c", 'cd "$1" && exec "$2"', "sh", folder, exe], {})
    raise OSError("no terminal program was found to run Codex in")


def open_cli(folder):
    """Start Codex in a new terminal window, working in a folder.

    Args: folder is the source folder.
    Returns: The codex command's path when it was started, or None when Codex
        is not installed. Raises OSError if the terminal cannot be started.
    """
    exe = find_cli()
    if exe is None:
        return None
    command, options = terminal_command(exe, folder)
    subprocess.Popen(command, **options)
    return exe


def send_command(platform=None):
    """Write the command Codex runs to send changes.

    Args: platform is a sys.platform value; None uses the running one.
    Returns: The command line: PowerShell syntax on Windows (where Codex
        runs PowerShell), POSIX shell syntax elsewhere.
    """
    exe = os.path.abspath(sys.executable)
    if (platform or sys.platform) == "win32":
        return f'& "{exe}" --send-changes'
    return f"{shlex.quote(exe)} --send-changes"


def guide_text(state):
    """Write the instructions Codex reads in a downloaded copy.

    Args: state is the copy's source_code.read_state() dict.
    Returns: The Markdown text.
    """
    build = state.get("build")
    return GUIDE.format(build=f"build {build}" if build else "a prebuilt copy",
                        commit=state["commit"][:7], command=send_command(),
                        request=source_code.REQUEST_FILE)


def write_guide(folder):
    """Put the instructions for Codex into a downloaded copy.

    Args: folder is the source folder Codex will open.
    Returns: The guide's path, or None for a source checkout (which keeps the
        developers' AGENTS.md). Raises OSError if it cannot be written.
    """
    state = source_code.read_state(folder)
    if state is None or updater.install_target() is None:
        return None
    path = os.path.join(folder, source_code.GUIDE_FILE)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(guide_text(state))
    return path
