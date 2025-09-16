.PHONY: backtest forecast clean

backtest:
	python -m src.backtest

forecast:
	python -m src.forecast

clean:
	rm -rf reports/*
