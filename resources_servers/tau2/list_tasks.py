"""List available tau2-bench domains and task IDs.

Usage:
    python resources_servers/tau2/list_tasks.py [domain]

Examples:
    python resources_servers/tau2/list_tasks.py           # list all domains
    python resources_servers/tau2/list_tasks.py airline    # list airline tasks
"""
import sys

from tau2.registry import registry


def main():
    if len(sys.argv) > 1:
        domain = sys.argv[1]
        tasks_loader = registry.get_tasks_loader(domain)
        tasks = tasks_loader("base")
        print(f"Domain: {domain}")
        print(f"Tasks ({len(tasks)}):")
        for task in tasks:
            scenario = str(task.user_scenario.instructions)[:80] if task.user_scenario else ""
            print(f"  {task.id}: {scenario}...")
    else:
        info = registry.get_info()
        print("Available domains:", info.domains)
        print("Available task sets:", info.task_sets)
        print("\nRun with a domain name to list tasks:")
        print("  python resources_servers/tau2/list_tasks.py airline")


if __name__ == "__main__":
    main()
