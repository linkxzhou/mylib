#!/bin/bash

# Concurrency Server Benchmark Test Script
# This script tests all 22 concurrency models using http_bench tool

# Configuration
SERVER_BINARY="../concurrency_server"
HTTP_BENCH="/Volumes/my/github/http_bench/http_bench"
SERVER_HOST="127.0.0.1"
SERVER_PORT="8000"
BENCH_CONNECTIONS="1000"
BENCH_DURATION="30s"
RESULT_DIR="benchmark_results"

# Docker Configuration
USE_DOCKER=false
DOCKER_IMAGE="concurrency-server:latest"
DOCKER_CONTAINER_PREFIX="concurrency-test"
DOCKER_BUILD_CONTEXT=".."

# Server models to test
declare -a MODELS=(
    "single_process"
    "multi_process"
    "multi_thread"
    "process_pool1"
    "process_pool2"
    "thread_pool"
    "leader_follower"
    "select"
    "poll"
    "epoll"
    "kqueue"
    "reactor"
    "coroutine"
    "event_loop"
    "work_stealing"
    "actor"
    "fiber"
    "producer_consumer"
    "half_sync_async"
    "proactor"
    "pipeline"
    "hybrid"
)

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo "Options:"
    echo "  -d, --docker          Use Docker containers for testing"
    echo "  -i, --image IMAGE     Docker image name (default: $DOCKER_IMAGE)"
    echo "  -c, --connections N   Number of connections (default: $BENCH_CONNECTIONS)"
    echo "  -t, --duration TIME   Test duration (default: $BENCH_DURATION)"
    echo "  -p, --port PORT       Server port (default: $SERVER_PORT)"
    echo "  -h, --help           Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                    # Run with local binary"
    echo "  $0 --docker           # Run with Docker containers"
    echo "  $0 -d -c 500 -t 60s   # Docker test with 500 connections for 60 seconds"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -i :$port >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to kill processes using the port
kill_port_processes() {
    local port=$1
    print_info "Checking for processes using port $port"
    
    local pids=$(lsof -ti :$port 2>/dev/null)
    if [ ! -z "$pids" ]; then
        print_warning "Found processes using port $port: $pids"
        for pid in $pids; do
            print_info "Killing process $pid"
            kill -9 $pid 2>/dev/null
        done
        sleep 2
    fi
}

# Function to build Docker image if needed
build_docker_image() {
    print_info "Checking Docker image: $DOCKER_IMAGE"
    
    if ! docker image inspect "$DOCKER_IMAGE" >/dev/null 2>&1; then
        print_info "Building Docker image: $DOCKER_IMAGE"
        if docker build -t "$DOCKER_IMAGE" "$DOCKER_BUILD_CONTEXT"; then
            print_success "Docker image built successfully: $DOCKER_IMAGE"
        else
            print_error "Failed to build Docker image: $DOCKER_IMAGE"
            return 1
        fi
    else
        print_info "Docker image already exists: $DOCKER_IMAGE"
    fi
}

# Function to stop and remove Docker container
stop_docker_container() {
    local container_name=$1
    
    if [ ! -z "$container_name" ]; then
        print_info "Stopping Docker container: $container_name"
        docker stop "$container_name" >/dev/null 2>&1 || true
        docker rm "$container_name" >/dev/null 2>&1 || true
    fi
}

# Function to wait for server to start
wait_for_server() {
    local max_attempts=30
    local attempt=1
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "http://${SERVER_HOST}:${SERVER_PORT}/" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    return 1
}

# Function to stop server (local binary)
stop_server() {
    if [ ! -z "$SERVER_PID" ]; then
        print_info "Stopping server (PID: $SERVER_PID)"
        kill $SERVER_PID 2>/dev/null
        sleep 1
        # Force kill if still running
        if kill -0 $SERVER_PID 2>/dev/null; then
            print_warning "Force killing server (PID: $SERVER_PID)"
            kill -9 $SERVER_PID 2>/dev/null
        fi
        wait $SERVER_PID 2>/dev/null
        SERVER_PID=""
    fi
    
    # Also clean up any remaining processes on the port
    kill_port_processes $SERVER_PORT
}

# Function to run benchmark for a specific model using Docker
run_docker_benchmark() {
    local model=$1
    local result_file="${RESULT_DIR}/${model}_docker_result.txt"
    local container_name="${DOCKER_CONTAINER_PREFIX}-${model}"
    
    print_info "Testing model: $model (Docker)"
    
    # Clean up any existing container
    stop_docker_container "$container_name"
    
    # Clean up any processes using the port
    kill_port_processes $SERVER_PORT
    
    # Start Docker container
    print_info "Starting Docker container: $container_name"
    
    if docker run -d --name "$container_name" -p "${SERVER_PORT}:8080" "$DOCKER_IMAGE" ./concurrency_server "$model" 8080 > "${RESULT_DIR}/${model}_docker.log" 2>&1; then
        print_success "Docker container started: $container_name"
        CURRENT_CONTAINER="$container_name"
    else
        print_error "Failed to start Docker container for model: $model"
        return 1
    fi
    
    # Wait for server to start
    if wait_for_server; then
        print_success "Server started successfully in container: $container_name"
    else
        print_error "Failed to start server for model: $model"
        stop_docker_container "$container_name"
        return 1
    fi
    
    # Run benchmark
    print_info "Running benchmark: $HTTP_BENCH -c $BENCH_CONNECTIONS -d $BENCH_DURATION 'http://${SERVER_HOST}:${SERVER_PORT}/'"
    
    echo "=== Docker Benchmark Results for $model ===" > "$result_file"
    echo "Date: $(date)" >> "$result_file"
    echo "Model: $model" >> "$result_file"
    echo "Runtime: Docker Container" >> "$result_file"
    echo "Container: $container_name" >> "$result_file"
    echo "Image: $DOCKER_IMAGE" >> "$result_file"
    echo "Connections: $BENCH_CONNECTIONS" >> "$result_file"
    echo "Duration: $BENCH_DURATION" >> "$result_file"
    echo "" >> "$result_file"
    
    sleep 10

    if $HTTP_BENCH -c $BENCH_CONNECTIONS -d $BENCH_DURATION "http://${SERVER_HOST}:${SERVER_PORT}/" >> "$result_file" 2>&1; then
        print_success "Benchmark completed for model: $model (Docker)"
    else
        print_error "Benchmark failed for model: $model (Docker)"
    fi
    
    # Get container logs
    docker logs "$container_name" > "${RESULT_DIR}/${model}_docker_server.log" 2>&1
    
    # Stop container
    stop_docker_container "$container_name"
    CURRENT_CONTAINER=""
    
    # Wait a bit before next test
    sleep 2
}

# Function to run benchmark for a specific model using local binary
run_local_benchmark() {
    local model=$1
    local result_file="${RESULT_DIR}/${model}_result.txt"
    
    print_info "Testing model: $model"
    
    # Clean up any processes using the port
    kill_port_processes $SERVER_PORT
    
    # Start server
    print_info "Starting server with model: $model"
    $SERVER_BINARY $model $SERVER_PORT > "${RESULT_DIR}/${model}_server.log" 2>&1 &
    SERVER_PID=$!
    
    # Wait for server to start
    if wait_for_server; then
        print_success "Server started successfully (PID: $SERVER_PID)"
    else
        print_error "Failed to start server for model: $model"
        stop_server
        return 1
    fi
    
    # Run benchmark
    print_info "Running benchmark: $HTTP_BENCH -c $BENCH_CONNECTIONS -d $BENCH_DURATION 'http://${SERVER_HOST}:${SERVER_PORT}/'"
    
    echo "=== Benchmark Results for $model ===" > "$result_file"
    echo "Date: $(date)" >> "$result_file"
    echo "Model: $model" >> "$result_file"
    echo "Connections: $BENCH_CONNECTIONS" >> "$result_file"
    echo "Duration: $BENCH_DURATION" >> "$result_file"
    echo "" >> "$result_file"
    
    sleep 10

    if $HTTP_BENCH -c $BENCH_CONNECTIONS -d $BENCH_DURATION "http://${SERVER_HOST}:${SERVER_PORT}/" >> "$result_file" 2>&1; then
        print_success "Benchmark completed for model: $model"
    else
        print_error "Benchmark failed for model: $model"
    fi
    
    # Stop server
    stop_server
    
    # Wait a bit before next test
    sleep 2
}

# Function to run benchmark (chooses Docker or local based on USE_DOCKER)
run_benchmark() {
    local model=$1
    
    if [ "$USE_DOCKER" = true ]; then
        run_docker_benchmark "$model"
    else
        run_local_benchmark "$model"
    fi
}

# Function to generate summary report
generate_summary() {
    local summary_file="${RESULT_DIR}/summary_report.txt"
    local runtime_suffix=""
    
    if [ "$USE_DOCKER" = true ]; then
        runtime_suffix="_docker"
    fi
    
    print_info "Generating summary report"
    
    echo "=== Concurrency Server Benchmark Summary ===" > "$summary_file"
    echo "Date: $(date)" >> "$summary_file"
    echo "Runtime: $([ "$USE_DOCKER" = true ] && echo "Docker Containers" || echo "Local Binary")" >> "$summary_file"
    if [ "$USE_DOCKER" = true ]; then
        echo "Docker Image: $DOCKER_IMAGE" >> "$summary_file"
    fi
    echo "Test Configuration:" >> "$summary_file"
    echo "  - Connections: $BENCH_CONNECTIONS" >> "$summary_file"
    echo "  - Duration: $BENCH_DURATION" >> "$summary_file"
    echo "  - Server: $SERVER_HOST:$SERVER_PORT" >> "$summary_file"
    echo "" >> "$summary_file"
    
    for model in "${MODELS[@]}"; do
        local result_file="${RESULT_DIR}/${model}${runtime_suffix}_result.txt"
        if [ -f "$result_file" ]; then
            echo "--- $model ---" >> "$summary_file"
            # Extract key metrics (this may need adjustment based on http_bench output format)
            grep -E "(Requests/sec|Total requests|Failed requests|Response time)" "$result_file" >> "$summary_file" 2>/dev/null || echo "No metrics found" >> "$summary_file"
            echo "" >> "$summary_file"
        fi
    done
    
    print_success "Summary report generated: $summary_file"
}

# Function to cleanup on exit
cleanup() {
    print_info "Cleaning up..."
    
    if [ "$USE_DOCKER" = true ]; then
        # Stop current container if running
        if [ ! -z "$CURRENT_CONTAINER" ]; then
            stop_docker_container "$CURRENT_CONTAINER"
        fi
        
        # Clean up any remaining containers
        for model in "${MODELS[@]}"; do
            stop_docker_container "${DOCKER_CONTAINER_PREFIX}-${model}"
        done
    else
        stop_server
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--docker)
            USE_DOCKER=true
            shift
            ;;
        -i|--image)
            DOCKER_IMAGE="$2"
            shift 2
            ;;
        -c|--connections)
            BENCH_CONNECTIONS="$2"
            shift 2
            ;;
        -t|--duration)
            BENCH_DURATION="$2"
            shift 2
            ;;
        -p|--port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Main execution
main() {
    print_info "Starting Concurrency Server Benchmark Test"
    print_info "Runtime: $([ "$USE_DOCKER" = true ] && echo "Docker Containers" || echo "Local Binary")"
    
    # Check prerequisites
    if [ "$USE_DOCKER" = true ]; then
        if ! command_exists docker; then
            print_error "Docker command not found. Please install Docker."
            exit 1
        fi
        
        # Build Docker image if needed
        if ! build_docker_image; then
            exit 1
        fi
    else
        if [ ! -f "$SERVER_BINARY" ]; then
            print_error "Server binary not found: $SERVER_BINARY"
            print_info "Please compile the server first: make"
            exit 1
        fi
    fi
    
    if [ ! -f "$HTTP_BENCH" ]; then
        print_error "http_bench tool not found: $HTTP_BENCH"
        print_info "Please ensure http_bench is available"
        exit 1
    fi
    
    if ! command_exists curl; then
        print_error "curl command not found. Please install curl."
        exit 1
    fi
    
    # Create result directory
    mkdir -p "$RESULT_DIR"
    
    # Clean up any existing processes on the test port
    print_info "Cleaning up any existing processes on port $SERVER_PORT"
    kill_port_processes $SERVER_PORT
    
    # Trap to ensure cleanup on script exit
    trap 'cleanup; exit' INT TERM EXIT
    
    print_info "Testing ${#MODELS[@]} concurrency models"
    print_info "Results will be saved in: $RESULT_DIR"
    
    # Run benchmarks for each model
    local successful_tests=0
    local total_tests=${#MODELS[@]}
    
    for model in "${MODELS[@]}"; do
        echo ""
        print_info "Progress: $((successful_tests + 1))/$total_tests"
        
        if run_benchmark "$model"; then
            ((successful_tests++))
        fi
    done
    
    echo ""
    print_info "Benchmark completed: $successful_tests/$total_tests tests successful"
    
    # Generate summary report
    generate_summary
    
    print_success "All benchmarks completed! Check results in: $RESULT_DIR"
}

# Run main function
main "$@"