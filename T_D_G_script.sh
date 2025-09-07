#!/bin/bash
cd /home/testuserclaw/claw/Textile_Design_Generator
source T_D_G_venv/bin/activate
uvicorn textile_api_server:app --host 0.0.0.0 --port 8000

