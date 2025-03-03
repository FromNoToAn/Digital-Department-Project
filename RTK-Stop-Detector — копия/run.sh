#!/bin/env sh

uvicorn base_detector:detector.app --reload --port 9004 --host 0.0.0.0 && bash
