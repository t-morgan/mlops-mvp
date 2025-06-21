import mlflow
import numpy as np
import pandas as pd  # Import pandas for the backtest summary
from common.spark_utils import get_spark_session
from common.mlflow_utils import setup_mlflow
from models.trading_env import TradingEnv
from models.dql_agent import DQLAgent

DELTA_TABLE_PATH = "s3a://delta/istanbul_stock"
MODEL_NAME = "stock_dql_agent"


def run_stock_dql_pipeline():
    setup_mlflow(experiment_name="stock_market_comparison")
    spark = get_spark_session("StockDQL")

    df_pandas = spark.read.format("delta").load(DELTA_TABLE_PATH).toPandas()
    df_pandas["date"] = pd.to_datetime(df_pandas["date"])
    df_pandas = df_pandas.set_index("date").sort_index()

    data = df_pandas.drop("target", axis=1).values
    train_size = int(len(data) * 0.8)
    train_data, test_data = data[:train_size], data[train_size:]

    env = TradingEnv(train_data)
    agent = DQLAgent(state_size=env.n_features)

    with mlflow.start_run(run_name="DQL_Agent_Training_and_Eval"):
        mlflow.log_param("model_type", "DQL")
        episodes = 10
        batch_size = 32

        print("--- Starting DQL Agent Training ---")
        for e in range(episodes):
            state = env.reset()
            state = np.reshape(state, [1, agent.state_size])
            done = False

            while not done:
                action = agent.act(state)
                next_state, reward, done = env.step(action)
                next_state = np.reshape(next_state, [1, agent.state_size])
                agent.remember(state, action, reward, next_state, done)
                state = next_state

            agent.replay(batch_size)
            print(
                f"Episode {e + 1}/{episodes}, Training Net Worth: {env.net_worth:.2f}"
            )
            mlflow.log_metric("training_net_worth", env.net_worth, step=e)

        print("\n--- Training Complete. Starting Back-Testing on Test Data ---")

        # Turn off exploration (epsilon=0) for pure exploitation/evaluation
        agent.epsilon = 0.0

        test_env = TradingEnv(test_data, initial_balance=env.initial_balance)
        state = test_env.reset()
        state = np.reshape(state, [1, agent.state_size])
        done = False

        while not done:
            action = agent.act(state)
            next_state, reward, done = test_env.step(action)
            state = np.reshape(next_state, [1, agent.state_size])

        # Log the final back-testing performance metrics to MLflow
        final_test_net_worth = test_env.net_worth
        total_profit = final_test_net_worth - test_env.initial_balance
        profit_percentage = (total_profit / test_env.initial_balance) * 100

        print(f"Back-Test Final Net Worth: {final_test_net_worth:.2f}")
        print(f"Back-Test Total Profit: {total_profit:.2f}")
        print(f"Back-Test Profit Percentage: {profit_percentage:.2f}%")

        mlflow.log_metric("backtest_final_net_worth", final_test_net_worth)
        mlflow.log_metric("backtest_profit_percentage", profit_percentage)

        # Log the model only after evaluation
        mlflow.keras.log_model(agent.model, "model", registered_model_name=MODEL_NAME)

    spark.stop()


if __name__ == "__main__":
    run_stock_dql_pipeline()
