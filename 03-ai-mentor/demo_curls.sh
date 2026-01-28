#!/bin/bash

# AI Mentor API - Demo curl commands
# Run the server first: python main.py

BASE_URL="http://localhost:8000"

echo "=== 1. Start a new mentoring session ==="
curl -X POST "$BASE_URL/sessions" \
  -H "Content-Type: application/json" | jq .

echo -e "\n\n=== 2. Send a message to session (replace SESSION_ID) ==="
curl -X POST "$BASE_URL/sessions/1/messages" \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello! I want to learn Python programming. Where should I start?"}' | jq .

echo -e "\n\n=== 3. Continue the conversation ==="
curl -X POST "$BASE_URL/sessions/1/messages" \
  -H "Content-Type: application/json" \
  -d '{"message": "I am a complete beginner with no programming experience."}' | jq .

echo -e "\n\n=== 4. Ask a follow-up question ==="
curl -X POST "$BASE_URL/sessions/1/messages" \
  -H "Content-Type: application/json" \
  -d '{"message": "What is a variable in Python?"}' | jq .

echo -e "\n\n=== 5. Get session details ==="
curl -X GET "$BASE_URL/sessions/1" \
  -H "Content-Type: application/json" | jq .

echo -e "\n\n=== 6. List all sessions ==="
curl -X GET "$BASE_URL/sessions" \
  -H "Content-Type: application/json" | jq .

echo -e "\n\n=== 7. End the session ==="
curl -X POST "$BASE_URL/sessions/1/end" \
  -H "Content-Type: application/json" | jq .

echo -e "\n\n=== Done! ==="
