#!/usr/bin/env python
"""Run all tests for nonconvexAG package."""

import sys
import subprocess
import argparse


def run_tests(args):
    """Run pytest with appropriate arguments."""
    
    # Base pytest command
    cmd = [sys.executable, "-m", "pytest"]
    
    # Add verbosity
    if args.verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")
        
    # Add coverage if requested
    if args.coverage:
        cmd.extend([
            "--cov=nonconvexAG",
            "--cov-report=term-missing",
            "--cov-report=html",
            "--cov-branch"
        ])
        
    # Add specific tests or markers
    if args.unit:
        cmd.append("tests/unit")
    elif args.integration:
        cmd.append("tests/integration")
    elif args.slow:
        cmd.extend(["-m", "slow"])
    elif args.fast:
        cmd.extend(["-m", "not slow"])
        
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])
        
    # Add any additional pytest args
    if args.pytest_args:
        cmd.extend(args.pytest_args)
        
    # Run the tests
    print(f"Running: {' '.join(cmd)}")
    return subprocess.call(cmd)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run tests for nonconvexAG package",
        epilog="Any additional arguments are passed directly to pytest"
    )
    
    # Test selection
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--unit", 
        action="store_true",
        help="Run only unit tests"
    )
    group.add_argument(
        "--integration",
        action="store_true", 
        help="Run only integration tests"
    )
    group.add_argument(
        "--slow",
        action="store_true",
        help="Run only slow tests (performance/stress tests)"
    )
    group.add_argument(
        "--fast",
        action="store_true",
        help="Skip slow tests (default)"
    )
    
    # Options
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "-n", "--parallel",
        type=int,
        metavar="N",
        help="Run tests in parallel with N workers"
    )
    
    # Collect remaining args for pytest
    args, pytest_args = parser.parse_known_args()
    args.pytest_args = pytest_args
    
    # Run tests
    sys.exit(run_tests(args))


if __name__ == "__main__":
    main()