curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-31b",
    "messages": [
      {"role": "user", "content": "My name is Alex. Remember that."},
      {"role": "assistant", "content": "Got it! Hello Alex."},
      {"role": "user", "content": "What is my name?"}
    ],
    "max_tokens": 50
  }' | python3 -m json.tool

curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-31b",
    "messages": [{"role": "user", "content": "Explain how transformers work in deep learning in 3 sentences."}],
    "max_tokens": 200,
    "temperature": 0.7
  }' | python3 -m json.tool

#vison
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-31b",
    "messages": [{
      "role": "user",
      "content": [
        {"type": "image_url", "image_url": {"url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3a/Cat03.jpg/1200px-Cat03.jpg"}},
        {"type": "text", "text": "What animal is in this image? Describe it briefly."}
      ]
    }],
    "max_tokens": 100
  }' | python3 -m json.tool

# streaming responce
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4-31b",
    "messages": [{"role": "user", "content": "Write a haiku about AI."}],
    "max_tokens": 100,
    "stream": true
  }'

# through put bench mark
for i in {1..5}; do
  START=$(date +%s%3N)
  RESULT=$(curl -s http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model":"gemma4-31b","messages":[{"role":"user","content":"Write a 100 word paragraph about space exploration."}],"max_tokens":150}')
  END=$(date +%s%3N)
  ELAPSED=$(( END - START ))
  TOKENS=$(echo $RESULT | python3 -c "import sys,json; r=json.load(sys.stdin); print(r['usage']['completion_tokens'])")
  TPS=$(echo "scale=1; $TOKENS * 1000 / $ELAPSED" | bc)
  echo "Run $i: ${TOKENS} tokens in ${ELAPSED}ms = ${TPS} tokens/sec"
done

