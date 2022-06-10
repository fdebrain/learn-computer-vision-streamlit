SHELL := /bin/bash

install:
	poetry install

run:
	streamlit run app.py