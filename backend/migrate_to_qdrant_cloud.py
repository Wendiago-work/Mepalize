#!/usr/bin/env python3
"""
Migration script to transfer data from local Qdrant to Qdrant Cloud
Uses the official Qdrant Migration Tool via Docker
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional

def run_command(cmd: list, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result"""
    print(f"Running: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        if result.stdout:
            print("STDOUT:", result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed with exit code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        raise

def check_docker_available() -> bool:
    """Check if Docker is available and running"""
    try:
        run_command(["docker", "--version"])
        run_command(["docker", "info"], check=False)
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Docker is not available or not running")
        return False

def pull_migration_tool() -> bool:
    """Pull the Qdrant migration tool Docker image"""
    try:
        print("📦 Pulling Qdrant migration tool...")
        run_command([
            "docker", "pull", 
            "registry.cloud.qdrant.io/library/qdrant-migration"
        ])
        print("✅ Migration tool pulled successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to pull migration tool: {e}")
        return False

def migrate_qdrant_to_cloud(
    source_host: str = "localhost",
    source_port: int = 6333,
    source_collection: str = "translation_embeddings",
    target_url: str = None,
    target_api_key: str = None,
    target_collection: str = None,
    batch_size: int = 64
) -> bool:
    """Migrate data from local Qdrant to Qdrant Cloud"""
    
    if not target_url or not target_api_key:
        print("❌ Target URL and API key are required")
        return False
    
    if not target_collection:
        target_collection = source_collection
    
    # Convert URLs to GRPC format for migration tool
    # Use host.docker.internal for Docker Desktop compatibility
    if source_host in ["localhost", "127.0.0.1"]:
        source_grpc_url = "http://host.docker.internal:6334"
    else:
        source_grpc_url = f"http://{source_host}:6334"  # Use GRPC port for source
    
    if target_url.startswith("https://"):
        target_grpc_url = target_url + ":6334"  # Add GRPC port for target
    else:
        target_grpc_url = target_url + ":6334"
    
    print(f"🚀 Starting migration from local Qdrant to Qdrant Cloud")
    print(f"   Source: {source_grpc_url}/{source_collection}")
    print(f"   Target: {target_grpc_url}/{target_collection}")
    
    try:
        cmd = [
            "docker", "run", "--rm", "-it",
            "registry.cloud.qdrant.io/library/qdrant-migration",
            "qdrant",
            "--source.url", source_grpc_url,
            "--source.collection", source_collection,
            "--target.url", target_grpc_url,
            "--target.api-key", target_api_key,
            "--target.collection", target_collection,
            "--migration.batch-size", str(batch_size)
        ]
        
        # Log the command before execution
        print(f"\n📋 Executing command:")
        print("docker run --rm -it \\")
        print("    registry.cloud.qdrant.io/library/qdrant-migration qdrant \\")
        print(f"    --source.url '{source_grpc_url}' \\")
        print(f"    --source.collection '{source_collection}' \\")
        print(f"    --target.url '{target_grpc_url}' \\")
        print(f"    --target.api-key '{target_api_key}' \\")
        print(f"    --target.collection '{target_collection}' \\")
        print(f"    --migration.batch-size {batch_size}")
        print()
        
        run_command(cmd)
        print("✅ Migration completed successfully!")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Migration failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Migrate data from local Qdrant to Qdrant Cloud"
    )
    parser.add_argument(
        "--source-host", 
        default="localhost",
        help="Source Qdrant host (default: localhost)"
    )
    parser.add_argument(
        "--source-port", 
        type=int,
        default=6333,
        help="Source Qdrant port (default: 6333)"
    )
    parser.add_argument(
        "--source-collection",
        default="translation_embeddings",
        help="Source collection name (default: translation_embeddings)"
    )
    parser.add_argument(
        "--target-url",
        required=True,
        help="Target Qdrant Cloud URL (e.g., https://your-cluster.eu-west-1-0.aws.cloud.qdrant.io)"
    )
    parser.add_argument(
        "--target-api-key",
        required=True,
        help="Target Qdrant Cloud API key"
    )
    parser.add_argument(
        "--target-collection",
        help="Target collection name (defaults to source collection name)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Migration batch size (default: 64)"
    )
    parser.add_argument(
        "--skip-docker-check",
        action="store_true",
        help="Skip Docker availability check"
    )
    
    args = parser.parse_args()
    
    print("🔄 Qdrant Cloud Migration Tool")
    print("=" * 50)
    
    # Check Docker availability
    if not args.skip_docker_check and not check_docker_available():
        sys.exit(1)
    
    # Pull migration tool
    if not pull_migration_tool():
        sys.exit(1)
    
    # Perform migration
    success = migrate_qdrant_to_cloud(
        source_host=args.source_host,
        source_port=args.source_port,
        source_collection=args.source_collection,
        target_url=args.target_url,
        target_api_key=args.target_api_key,
        target_collection=args.target_collection,
        batch_size=args.batch_size
    )
    
    if success:
        print("\n🎉 Migration completed successfully!")
        print("You can now update your environment variables to use Qdrant Cloud:")
        print(f"   TRANSLATION_USE_QDRANT_CLOUD=true")
        print(f"   TRANSLATION_QDRANT_CLOUD_URL={args.target_url}")
        print(f"   TRANSLATION_QDRANT_CLOUD_API_KEY={args.target_api_key}")
        sys.exit(0)
    else:
        print("\n❌ Migration failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
