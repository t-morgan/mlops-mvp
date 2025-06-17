#!/bin/bash
set -e

function create_user_and_database() {
    local db_name=$1
    local user_name=$2
    local password=$3
    
    # Check if user exists
    if psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -tAc "SELECT 1 FROM pg_roles WHERE rolname='$user_name'" | grep -q 1; then
        echo "User '$user_name' already exists, skipping creation."
    else
        echo "Creating user '$user_name'."
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -c "CREATE USER $user_name WITH PASSWORD '$password';"
    fi

    # Check if database exists
    if psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -tAc "SELECT 1 FROM pg_database WHERE datname='$db_name'" | grep -q 1; then
        echo "Database '$db_name' already exists, skipping creation."
    else
        echo "Creating database '$db_name'."
        psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -c "CREATE DATABASE $db_name;"
    fi

    echo "Granting all privileges on database '$db_name' to user '$user_name'."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" -c "GRANT ALL PRIVILEGES ON DATABASE $db_name TO $user_name;"
}

if [ -n "$POSTGRES_MULTIPLE_DATABASES" ]; then
    echo "Multiple database creation requested"
    # Create the mlflow user and DB (from the POSTGRES_DB environment variable)
    create_user_and_database "$POSTGRES_DB" "$POSTGRES_USER" "$POSTGRES_PASSWORD"
    
    # Create the airflow user and DB
    if [ -n "$POSTGRES_AIRFLOW_USER" ]; then
        create_user_and_database "airflow" "$POSTGRES_AIRFLOW_USER" "$POSTGRES_AIRFLOW_PASSWORD"
    fi
    echo "Multiple databases creation finished"
fi