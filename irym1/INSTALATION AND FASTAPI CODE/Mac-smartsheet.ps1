#!/bin/bash

# ===== Colors =====
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
MAGENTA='\033[0;35m'
RESET='\033[0m'

CONFIG_FILE="$HOME/.irym_config"
ENV_NAME="irym_1"
ENV_FILE="environment.yml"

# ===== Check Python =====
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python3 not found! Install it first.${RESET}"
    exit 1
else
    echo -e "${GREEN}Python3 found.${RESET}"
fi

# ===== Check Conda =====
if ! command -v conda &> /dev/null; then
    echo -e "${YELLOW}Conda not found. Installing Miniconda...${RESET}"
    curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-x86_64.sh
    bash Miniconda3-latest-MacOSX-x86_64.sh -b -p $HOME/miniconda3
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    echo -e "${GREEN}Miniconda installed.${RESET}"
else
    source ~/anaconda3/etc/profile.d/conda.sh
    echo -e "${GREEN}Conda found.${RESET}"
fi

# ===== Ask Project Path =====
if [ -f "$CONFIG_FILE" ]; then
    PROJECT_DIR=$(grep "PROJECT_DIR" "$CONFIG_FILE" | cut -d'=' -f2)
    if [ ! -d "$PROJECT_DIR" ]; then
        read -p "Saved project path invalid. Enter correct path: " PROJECT_DIR
        echo "PROJECT_DIR=$PROJECT_DIR" > "$CONFIG_FILE"
    fi
else
    read -p "Enter your project path: " PROJECT_DIR
    echo "PROJECT_DIR=$PROJECT_DIR" > "$CONFIG_FILE"
fi

cd "$PROJECT_DIR" || { echo -e "${RED}Cannot go to project folder${RESET}"; exit 1; }

# ===== Conda Environment =====
if conda env list | grep -q "$ENV_NAME"; then
    echo -e "${GREEN}Activating $ENV_NAME...${RESET}"
    conda activate "$ENV_NAME"
elif [ -f "$PROJECT_DIR/$ENV_FILE" ]; then
    echo -e "${YELLOW}Creating environment $ENV_NAME...${RESET}"
    conda env create -f "$ENV_FILE"
    conda activate "$ENV_NAME"
else
    echo -e "${RED}Environment file $ENV_FILE not found.${RESET}"
    exit 1
fi

# ===== Menu =====
while true; do
    echo
    echo -e "${CYAN}===== IRYM 1 COMMAND CENTER =====${RESET}"
    echo -e "${BLUE}[R]${RESET} Run server"
    echo -e "${BLUE}[M]${RESET} Make & apply migrations"
    echo -e "${BLUE}[C]${RESET} Collect static files"
    echo -e "${BLUE}[K]${RESET} Set NG_KEY in .env"
    echo -e "${BLUE}[0]${RESET} Clear screen"
    echo -e "${BLUE}[Q]${RESET} Quit"
    echo -n -e "${MAGENTA}Choose an option: ${RESET}"

    read -n1 choice
    echo

    case "$choice" in
        [Rr])
            echo -e "${GREEN}Running Django server...${RESET}"
            python3 manage.py runserver
            ;;
        [Mm])
            echo -e "${GREEN}Making migrations...${RESET}"
            python3 manage.py makemigrations
            python3 manage.py migrate
            ;;
        [Cc])
            echo -e "${GREEN}Collecting static files...${RESET}"
            python3 manage.py collectstatic --noinput
            ;;
        [Kk])
            read -s -p "Enter NG_KEY: " ng_key_value
            echo
            ENV_FILE_PATH="$PROJECT_DIR/.env"
            if [ -f "$ENV_FILE_PATH" ]; then
                if grep -q "^NG_KEY=" "$ENV_FILE_PATH"; then
                    sed -i '' "s|^NG_KEY=.*|NG_KEY=$ng_key_value|" "$ENV_FILE_PATH"
                else
                    echo "NG_KEY=$ng_key_value" >> "$ENV_FILE_PATH"
                fi
            else
                echo "NG_KEY=$ng_key_value" > "$ENV_FILE_PATH"
            fi
            echo -e "${GREEN}NG_KEY saved in .env${RESET}"
            ;;
        [0])
            clear
            ;;
        [Qq])
            echo -e "${CYAN}Exiting...${RESET}"
            break
            ;;
        *)
            echo -e "${RED}Invalid option!${RESET}"
            ;;
    esac
done

