#!/usr/bin/env python3
"""Test Labelbox export with a small subset of data."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

import labelbox as lb
from cje.utils.progress import console

# Test API key and create a test dataset
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VySWQiOiJjbWMzbjM2Z3QwYTdmMDd6bTRxYm1jb2NlIiwib3JnYW5pemF0aW9uSWQiOiJjbWMzbjM2Z2swYTdlMDd6bWFqd2NkaDd0IiwiYXBpS2V5SWQiOiJjbWMzcW1yOHYwamd0MDd5NzIxNXZhdTg4Iiwic2VjcmV0IjoiNWQ0ZmIxZTVmZWQ4ODdiMGUzZTAwMGQ2ZjE1N2FlM2EiLCJpYXQiOjE3NTAzNTkxMjgsImV4cCI6MTc1Mjk1MTEyOH0.AkwXYjY3CMRo7bhlWM_vdnnnVBEP5sVS4xmLJ2YW0C0"

try:
    # Initialize client
    client = lb.Client(api_key=api_key)
    console.print("✅ Successfully connected to Labelbox")

    # Get organization info
    org = client.get_organization()
    console.print(f"📦 Organization: {org.name}")

    # Create a test dataset
    dataset_name = "CJE_Arena_Test_Dataset"
    dataset = client.create_dataset(name=dataset_name)
    console.print(f"✅ Created dataset: {dataset.name} (ID: {dataset.uid})")

    # Create a few test data rows
    test_assets = [
        {
            "row_data": "USER: What is 2+2?\n\nASSISTANT: 2+2 equals 4.",
            "global_key": "test_task_000001",
            "media_type": "TEXT",
            "metadata_fields": [
                {"name": "task_type", "value": "conversation_rating"},
                {"name": "prompt_length", "value": 15},
                {"name": "response_length", "value": 13},
            ],
        },
        {
            "row_data": "USER: Explain quantum physics\n\nASSISTANT: Quantum physics is the study of matter and energy at the smallest scales.",
            "global_key": "test_task_000002",
            "media_type": "TEXT",
            "metadata_fields": [
                {"name": "task_type", "value": "conversation_rating"},
                {"name": "prompt_length", "value": 25},
                {"name": "response_length", "value": 65},
            ],
        },
    ]

    # Upload test data
    console.print(f"\n📤 Uploading {len(test_assets)} test data rows...")
    task = dataset.create_data_rows(test_assets)
    task.wait_till_done()

    if task.errors:
        console.print("[red]Upload errors:[/red]")
        for error in task.errors:
            console.print(f"   • {error}")
    else:
        console.print("[green]✓ Test data uploaded successfully[/green]")

    console.print(f"\n✅ Test complete! Dataset ID: {dataset.uid}")
    console.print("\nNow you can run the full export with:")
    console.print(
        f"python 03b_export_for_labelbox.py --api-key {api_key} --dataset-id {dataset.uid}"
    )

except Exception as e:
    console.print(f"[red]Error: {e}[/red]")
    import traceback

    traceback.print_exc()
