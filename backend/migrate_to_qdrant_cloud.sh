#!/bin/bash

# Qdrant Cloud Migration Script (reads from .env file)
# This script migrates data from local Qdrant to Qdrant Cloud using .env file

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}ℹ️  $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_error() {
    echo -e "${RED}❌ $1${NC}"
}

# Function to load environment variables from .env file
load_env() {
    if [[ -f .env ]]; then
        print_info "Loading environment variables from .env file..."
        export $(grep -v '^#' .env | grep -v '^$' | xargs)
        print_success "Environment variables loaded"
    else
        print_error ".env file not found"
        exit 1
    fi
}

# Function to check required environment variables
check_env_vars() {
    local missing_vars=()
    
    if [[ -z "$TRANSLATION_QDRANT_CLOUD_URL" ]]; then
        missing_vars+=("TRANSLATION_QDRANT_CLOUD_URL")
    fi
    
    if [[ -z "$TRANSLATION_QDRANT_CLOUD_API_KEY" ]]; then
        missing_vars+=("TRANSLATION_QDRANT_CLOUD_API_KEY")
    fi
    
    if [[ ${#missing_vars[@]} -gt 0 ]]; then
        print_error "Missing required environment variables:"
        for var in "${missing_vars[@]}"; do
            echo "  - $var"
        done
        exit 1
    fi
    
    print_success "All required environment variables are set"
}

# Function to run the migration
run_migration() {
    local source_host="${TRANSLATION_QDRANT_HOST:-localhost}"
    local source_port="${TRANSLATION_QDRANT_PORT:-6334}"  # Use GRPC port for migration
    local source_collection="${TRANSLATION_QDRANT_COLLECTION_NAME:-translation_embeddings}"
    local target_url="$TRANSLATION_QDRANT_CLOUD_URL"
    local target_api_key="$TRANSLATION_QDRANT_CLOUD_API_KEY"
    local target_collection="${TRANSLATION_QDRANT_COLLECTION_NAME:-translation_embeddings}"
    local batch_size="${MIGRATION_BATCH_SIZE:-64}"
    
    print_info "Starting Qdrant Cloud Migration"
    echo "=================================="
    print_info "Source: $source_host:$source_port/$source_collection"
    print_info "Target: $target_url/$target_collection"
    print_info "Batch size: $batch_size"
    echo ""
    
    # Check if Python script exists
    local script_path="./migrate_to_qdrant_cloud.py"
    if [[ ! -f "$script_path" ]]; then
        print_error "Migration script not found: $script_path"
        exit 1
    fi
    
    # Make sure the script is executable
    chmod +x "$script_path"
    
    # Run the migration
    print_info "Running migration..."
    python3 "$script_path" \
        --source-host "$source_host" \
        --source-port "$source_port" \
        --source-collection "$source_collection" \
        --target-url "$target_url" \
        --target-api-key "$target_api_key" \
        --target-collection "$target_collection" \
        --batch-size "$batch_size"
    
    if [[ $? -eq 0 ]]; then
        print_success "Migration completed successfully!"
        echo ""
        print_info "Next steps:"
        echo "1. Set TRANSLATION_USE_QDRANT_CLOUD=true in your .env file"
        echo "2. Restart your application to use Qdrant Cloud"
        echo ""
        print_success "Migration complete! 🎉"
    else
        print_error "Migration failed!"
        exit 1
    fi
}

# Main execution
main() {
    print_info "🔄 Qdrant Cloud Migration Tool (from .env)"
    print_info "=========================================="
    echo ""
    
    # Load environment variables
    load_env
    
    # Check required variables
    check_env_vars
    
    # Run migration
    run_migration
}

# Run main function
main "$@"
