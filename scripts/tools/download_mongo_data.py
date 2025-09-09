#!/usr/bin/env python3

"""
MongoDB Data Export Utility

This module provides functionality to download and export data from MongoDB databases.
It connects to a specified MongoDB cluster, retrieves all collections from a given
database, and saves each collection as a separate JSON file.

Features:
- Export entire database collections to JSON format
- Automatic handling of MongoDB ObjectId serialization
- Organized output with timestamped directories
- Simple command-line interface for direct usage

Requirements:
- pymongo: For MongoDB connectivity
- json: For JSON serialization
- datetime: For timestamp generation
- os: For directory operations

Usage:
    from mongodb_export import download_mongodb_data
    
    # Download all data from a specific database
    download_mongodb_data(
        connection_string="mongodb+srv://username:password@cluster.mongodb.net/dbname",
        database_name="your_database_name",
        output_dir="optional_custom_directory"
    )

Author: Liam Barrett
Version: 1.0.0
Date: 04/03/2025
"""

import json
from datetime import datetime, date
import os
import pymongo

def download_mongodb_data(connection_string, database_name, output_dir=None):
    """
    Download all collections from a MongoDB database and save them as JSON files.
    
    Args:
        connection_string (str): MongoDB connection string
        database_name (str): Name of the database to download
        output_dir (str, optional): Directory to save output files. Defaults to current date.
    """
    # Connect to MongoDB
    client = pymongo.MongoClient(connection_string)
    db = client[database_name]

    # Create output directory if not specified
    if output_dir is None:
        output_dir = f"mongodb_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get all collection names
    collections = db.list_collection_names()

    print(f"Found {len(collections)} collections in database '{database_name}'")

    # Download each collection
    for collection_name in collections:
        collection = db[collection_name]

        # Get all documents in the collection
        documents = list(collection.find({}))

        # Convert ObjectId to string for JSON serialization
        for doc in documents:
            if '_id' in doc and not isinstance(doc['_id'], str):
                doc['_id'] = str(doc['_id'])

        # Save to file
        output_file = os.path.join(output_dir, f"{collection_name}.json")
        with open(output_file, 'w') as f:
            json.dump(documents, f, default=str, indent=2)

        print(f"Exported {len(documents)} documents from collection '{collection_name}' to {output_file}")

    print(f"Export complete. All data saved to directory: {output_dir}")
    client.close()

# Example usage
if __name__ == "__main__":
    CONNECTION_STRING = "mongodb+srv://lbarrett16:vt9WpBoifAjZR0fM@annotations.mhup2.mongodb.net/?retryWrites=true&w=majority&appName=Annotations"
    DATABASE_NAME = "clinical-annotations"
    OUTPUT_PATH = f"./nlp_ehr/results/{date.today()}"

    download_mongodb_data(CONNECTION_STRING, DATABASE_NAME)
