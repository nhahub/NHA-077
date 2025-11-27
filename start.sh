#!/bin/bash
uvicorn API.server:app --host 0.0.0.0 --port ${PORT}