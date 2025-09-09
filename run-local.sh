#!/bin/bash
# Run both frontend and backend LocalTunnel tunnels

# Frontend
FRONTEND_PORT=5173
FRONTEND_SUBDOMAIN=frontendmepalize

# Backend
BACKEND_PORT=8000
BACKEND_SUBDOMAIN=backendmepalize

# Start frontend tunnel in background
echo "Starting frontend tunnel on port $FRONTEND_PORT..."
nohup lt --port $FRONTEND_PORT --subdomain $FRONTEND_SUBDOMAIN > frontend.log 2>&1 &

# Start backend tunnel in background
echo "Starting backend tunnel on port $BACKEND_PORT..."
nohup lt --port $BACKEND_PORT --subdomain $BACKEND_SUBDOMAIN > backend.log 2>&1 &

echo "Tunnels started!"
echo "Frontend URL: https://$FRONTEND_SUBDOMAIN.loca.lt"
echo "Backend URL:  https://$BACKEND_SUBDOMAIN.loca.lt"
