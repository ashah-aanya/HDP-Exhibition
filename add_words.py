#!/usr/bin/env python3
"""
Interactive script to add words to the visualization
Run this while the server is running to add words interactively
"""
import asyncio
import websockets
import json
import sys


async def add_words_interactive():
    """Connect to WebSocket server and add words interactively"""
    uri = "ws://localhost:8080/ws"

    print("=" * 60)
    print("Interactive Word Addition Tool (Debug Mode)")
    print("=" * 60)
    print(f"[DEBUG] Attempting to connect to: {uri}")

    try:
        async with websockets.connect(uri) as websocket:
            print("[DEBUG] WebSocket connection established")
            print("✓ Connected successfully!")
            print("\nType words and press Enter to add them to the visualization.")
            print("New words will appear orange and turn blue after 5 seconds.")
            print("Type 'stop' to exit.\n")

            # Receive initial state
            print("[DEBUG] Waiting for initial state from server...")
            initial_msg = await websocket.recv()
            print(f"[DEBUG] Received initial message: {initial_msg[:100]}...")
            initial_data = json.loads(initial_msg)
            print(f"[DEBUG] Initial state type: {initial_data.get('type')}")
            print(f"✓ Current visualization has {len(initial_data.get('words', []))} words.\n")

            word_count = 0
            while True:
                # Get input from user
                word = input("Enter a word: ").strip()

                if not word:
                    print("[DEBUG] Empty input, skipping...")
                    continue

                if word.lower() == 'stop':
                    print("\n[DEBUG] Stop command received")
                    print("Stopping word addition. Goodbye!")
                    break

                word_count += 1
                print(f"[DEBUG] Processing word #{word_count}: '{word}'")

                # Send command to add word
                command = {
                    "command": "add_word",
                    "word": word
                }
                print(f"[DEBUG] Sending command: {json.dumps(command)}")
                await websocket.send(json.dumps(command))
                print("[DEBUG] Command sent, waiting for response...")

                # Wait for response and any updates
                messages_received = 0
                try:
                    # Listen for multiple messages (response + updates)
                    while messages_received < 3:  # Response + potential updates
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=1.0)
                            messages_received += 1
                            response_data = json.loads(response)
                            msg_type = response_data.get('type')

                            print(f"[DEBUG] Received message #{messages_received}, type: {msg_type}")

                            if msg_type == 'response':
                                if response_data.get('success'):
                                    print(f"✓ Server confirmed: '{word}' added successfully")
                                    print(f"  → Will appear ORANGE, then turn BLUE in 5 seconds")
                                else:
                                    print(f"✗ Server response: {response_data.get('message', 'Failed to add word')}")
                            elif msg_type == 'update':
                                word_list = response_data.get('words', [])
                                print(f"[DEBUG] Visualization updated with {len(word_list)} total words")
                        except asyncio.TimeoutError:
                            print("[DEBUG] No more messages (timeout)")
                            break

                    if messages_received == 0:
                        print(f"⚠ No response received, but '{word}' was sent to server")

                except Exception as e:
                    print(f"[DEBUG] Error receiving response: {e}")

                print()

    except ConnectionRefusedError:
        print("\n[DEBUG] Connection refused")
        print("✗ Error: Could not connect to server.")
        print("Make sure the server is running with: python realtime_server.py")
        print("\nTo start the server:")
        print("  Terminal 1: source venv/bin/activate && python realtime_server.py")
        print("  Terminal 2: cd /Users/aanyashah/HDP-Exhibition && python -m http.server 8000")
    except websockets.exceptions.InvalidURI:
        print(f"\n[DEBUG] Invalid URI: {uri}")
        print("✗ Error: Invalid WebSocket URI")
    except Exception as e:
        print(f"\n[DEBUG] Unexpected error: {type(e).__name__}")
        print(f"✗ Error: {e}")
        import traceback
        print("\n[DEBUG] Full traceback:")
        traceback.print_exc()


def main():
    """Run the interactive word addition"""
    try:
        asyncio.run(add_words_interactive())
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye!")


if __name__ == "__main__":
    main()
