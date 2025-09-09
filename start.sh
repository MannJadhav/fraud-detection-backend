#!/bin/bash
gunicorn -k uvicorn.workers.UvicornWorker app.app:app --bind=0.0.0.0:$PORT
