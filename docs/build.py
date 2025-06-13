#!/usr/bin/env python3
"""
Sphinx documentation build script with auto-rebuild capabilities.

This script provides commands for building, serving, and auto-rebuilding
the CJE-Core documentation.
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Callable, Any

import typer
from watchdog.events import FileSystemEventHandler, FileSystemEvent
from watchdog.observers import Observer

app = typer.Typer(help="CJE-Core documentation build tools")

# Configuration
DOCS_DIR = Path(__file__).parent
PROJECT_ROOT = DOCS_DIR.parent
BUILD_DIR = DOCS_DIR / "_build"
SOURCE_DIR = DOCS_DIR
WATCH_DIRS = [
    PROJECT_ROOT / "cje",  # Source code for autodoc
    DOCS_DIR,  # Documentation sources
]


class DocEventHandler(FileSystemEventHandler):
    """Handler for documentation file changes."""

    def __init__(self, build_func: Callable[[], None]) -> None:
        self.build_func = build_func
        self.last_build: float = 0
        self.debounce_delay: float = 2  # seconds

    def on_modified(self, event: FileSystemEvent) -> None:
        if event.is_directory:
            return

        # Debounce rapid file changes
        now = time.time()
        if now - self.last_build < self.debounce_delay:
            return

        # Ensure src_path is a string
        src_path = str(event.src_path)
        if any(src_path.endswith(ext) for ext in [".py", ".rst", ".md"]):
            typer.echo(f"🔄 File changed: {src_path}")
            self.build_func()
            self.last_build = now


def run_sphinx_command(
    builder: str = "html", extra_args: Optional[List[str]] = None
) -> bool:
    """Run sphinx-build command."""
    cmd = [
        "sphinx-build",
        "-b",
        builder,
        # "-W",  # Temporarily disable treating warnings as errors
        str(SOURCE_DIR),
        str(BUILD_DIR / builder),
    ]

    if extra_args:
        cmd.extend(extra_args)

    typer.echo(f"🔨 Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd, cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        )
        typer.echo("✅ Build successful!")
        return True
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Build failed with exit code {e.returncode}", err=True)
        typer.echo(f"STDOUT:\n{e.stdout}", err=True)
        typer.echo(f"STDERR:\n{e.stderr}", err=True)
        return False


@app.command()
def build(
    builder: str = typer.Option("html", help="Sphinx builder to use"),
    clean: bool = typer.Option(
        False, "--clean", "-c", help="Clean build directory first"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """Build the documentation."""
    if clean:
        typer.echo("🧹 Cleaning build directory...")
        import shutil

        if BUILD_DIR.exists():
            shutil.rmtree(BUILD_DIR)

    extra_args = []
    if verbose:
        extra_args.append("-v")

    success = run_sphinx_command(builder, extra_args)

    if success and builder == "html":
        index_file = BUILD_DIR / "html" / "index.html"
        if index_file.exists():
            typer.echo(f"📖 Documentation built: file://{index_file.absolute()}")

    sys.exit(0 if success else 1)


@app.command()
def serve(
    port: int = typer.Option(8000, help="Port to serve on"),
    host: str = typer.Option("localhost", help="Host to serve on"),
) -> None:
    """Serve the built documentation."""
    html_dir = BUILD_DIR / "html"

    if not html_dir.exists():
        typer.echo("❌ HTML documentation not found. Run 'build' first.")
        raise typer.Exit(1)

    try:
        import http.server
        import socketserver

        os.chdir(html_dir)

        with socketserver.TCPServer(
            (host, port), http.server.SimpleHTTPRequestHandler
        ) as httpd:
            typer.echo(f"🌐 Serving documentation at http://{host}:{port}")
            typer.echo("Press Ctrl+C to stop")
            httpd.serve_forever()

    except KeyboardInterrupt:
        typer.echo("\n👋 Server stopped")
    except Exception as e:
        typer.echo(f"❌ Failed to start server: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def watch(
    port: int = typer.Option(8000, help="Port to serve on"),
    host: str = typer.Option("localhost", help="Host to serve on"),
) -> None:
    """Watch for changes and auto-rebuild documentation."""
    # Initial build
    typer.echo("🔨 Initial build...")
    if not run_sphinx_command():
        typer.echo("❌ Initial build failed")
        raise typer.Exit(1)

    # Start file watcher
    def build_docs() -> None:
        run_sphinx_command()

    event_handler = DocEventHandler(build_docs)
    observer = Observer()

    for watch_dir in WATCH_DIRS:
        if watch_dir.exists():
            observer.schedule(event_handler, str(watch_dir), recursive=True)
            typer.echo(f"👀 Watching: {watch_dir}")

    observer.start()

    # Start server in background
    try:
        import threading
        import http.server
        import socketserver

        html_dir = BUILD_DIR / "html"
        os.chdir(html_dir)

        with socketserver.TCPServer(
            (host, port), http.server.SimpleHTTPRequestHandler
        ) as httpd:
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()

            typer.echo(f"🌐 Serving at http://{host}:{port}")
            typer.echo("👀 Watching for changes... Press Ctrl+C to stop")

            while True:
                time.sleep(1)

    except KeyboardInterrupt:
        typer.echo("\n🛑 Stopping watcher and server...")
        observer.stop()

    observer.join()


@app.command()
def linkcheck() -> None:
    """Check external links in documentation."""
    typer.echo("🔗 Checking external links...")
    success = run_sphinx_command("linkcheck")

    if success:
        linkcheck_file = BUILD_DIR / "linkcheck" / "output.txt"
        if linkcheck_file.exists():
            typer.echo(f"📝 Link check results: {linkcheck_file.absolute()}")

    sys.exit(0 if success else 1)


@app.command()
def pdf() -> None:
    """Build PDF documentation."""
    typer.echo("📄 Building PDF documentation...")
    success = run_sphinx_command("latexpdf")

    if success:
        pdf_file = BUILD_DIR / "latex" / "CJE-Core.pdf"
        if pdf_file.exists():
            typer.echo(f"📄 PDF built: {pdf_file.absolute()}")

    sys.exit(0 if success else 1)


@app.command()
def autodoc() -> None:
    """Generate autodoc stub files for all modules."""
    typer.echo("📝 Generating autodoc stubs...")

    cmd = [
        "sphinx-apidoc",
        "-o",
        str(DOCS_DIR / "api" / "generated"),
        "-f",  # Force overwrite
        "-e",  # Put each module on separate page
        str(PROJECT_ROOT / "cje"),
        str(PROJECT_ROOT / "cje/testing"),  # Exclude testing
    ]

    try:
        subprocess.run(cmd, check=True)
        typer.echo("✅ Autodoc stubs generated!")
    except subprocess.CalledProcessError as e:
        typer.echo(f"❌ Failed to generate autodoc stubs: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
