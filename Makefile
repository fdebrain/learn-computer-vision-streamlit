SHELL := /bin/bash

install:
	poetry install

run:
	streamlit run welcome.py
