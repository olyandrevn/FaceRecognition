import pandas as pd
from sqlalchemy import create_engine


class Aggregator:
    """
    Aggregator class to combine customer appearance data from multiple store databases incrementally.

    Methods:
        add_store(store_name, db_url): Add a store database to the aggregator.
        aggregate_data(start_time, end_time): Aggregate appearance data across all stores.
    """

    def __init__(self):
        """
        Initialize the aggregator with an empty list of stores and cached results.
        """
        self.stores = {}
        self.aggregated_data = pd.DataFrame(columns=["customer_id", "total_appearances", "store_id", "timestamp"])  # Cached results

    def add_store(self, store):
        """
        Add a store database to the aggregator and incrementally update the cached results.

        Args:
            store_name (str): The name of the store (e.g., "store_001").
            db_url (str): The database URL for the store's database.
        """
        if store["store_id"] in self.stores:
            print(f"Store '{store["store_id"]}' already added.")
            return

        # Add the store
        self.stores[store["store_id"]] = create_engine(store["store_db"])

        # Incrementally aggregate data for the new store
        print(f"Aggregating data for new store: {store["store_id"]}")
        new_store_data = self._aggregate_store_data(store["store_id"])
        self.aggregated_data = (
            pd.concat([self.aggregated_data, new_store_data])
            .groupby("customer_id", as_index=False)
            .sum()
        )

    def _aggregate_store_data(self, store_name, start_time=None, end_time=None) -> pd.DataFrame:
        """
        Aggregate appearance data for a single store within an optional timeframe.

        Args:
            store_name (str): The name of the store.
            start_time (str, optional): Start of the timeframe (ISO 8601 format).
            end_time (str, optional): End of the timeframe (ISO 8601 format).

        Returns:
            pandas.DataFrame: Aggregated appearance data for the store.
        """
        engine = self.stores[store_name]

        # Build the SQL query
        query = f"SELECT customer_id, COUNT(*) AS appearances FROM {store_name}_customers"
        conditions = []
        if start_time:
            conditions.append(f"timestamp >= '{start_time}'")
        if end_time:
            conditions.append(f"timestamp <= '{end_time}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " GROUP BY customer_id"

        # Query the store database
        store_data = pd.read_sql(query, con=engine)
        return store_data

    def aggregate_data(self, start_time=None, end_time=None):
        """
        Aggregate appearance data across all stores, optionally filtering by timeframe.

        Args:
            start_time (str, optional): Start of the timeframe (ISO 8601 format).
            end_time (str, optional): End of the timeframe (ISO 8601 format).

        Returns:
            pandas.DataFrame: Aggregated appearance data.
        """
        if start_time or end_time:
            print("Performing aggregation for the specified timeframe...")
            data_frames = [
                self._aggregate_store_data(store_name, start_time, end_time)
                for store_name in self.stores
            ]
            combined_data = pd.concat(data_frames, ignore_index=True)
            return combined_data.groupby("customer_id", as_index=False).sum()

        return self.aggregated_data
    

if __name__ == "__main__":
    config = Config()

    # Define store databases
    stores = [
        {
            "store_id": "store_001",
            "store_db": "sqlite:///store_001.db"
        },
        {
            "store_id": "store_002",
            "store_db": "sqlite:///store_002.db"
        },
    ]

    # Initialize the Aggregator
    aggregator = Aggregator()

    # Add store databases to the aggregator
    for store_data in stores:
        aggregator.add_store(store_data)

    # Aggregate data
    start_time, end_time = "2021-01-01T00:00:00", "2021-01-02T00:00:00"
    aggregated_data = aggregator.aggregate_data(start_time, end_time)
    print("Aggregated Customer Data:")
    print(aggregated_data)

