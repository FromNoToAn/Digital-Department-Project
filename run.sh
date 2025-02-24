#!/bin/env sh

uvicorn base_detector:detector.app --reload --port 8000 --host 0.0.0.0 && bash
