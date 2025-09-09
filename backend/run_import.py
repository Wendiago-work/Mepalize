#!/usr/bin/env python3
"""
Simple runner script for data import
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from import_data_to_mongodb import main

if __name__ == "__main__":
    print("🚀 Starting MongoDB Data Import...")
    print("=" * 50)
    
    try:
        asyncio.run(main())
        print("\n✅ Import completed successfully!")
    except KeyboardInterrupt:
        print("\n⏹️ Import cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Import failed: {e}")
        sys.exit(1)
