import pandas as pd

def parse_loss_file(filepath, output_excel):
    epochs = []
    train_losses = []
    val_losses = []
    
    with open(filepath, "r") as f:
        for line in f:
            line = line.strip()
            
            # Check for Epoch line
            if line.startswith("Epoch"):
                # Format is usually "Epoch 1/300", so split by space then '/'
                parts = line.split()
                epoch_str = parts[1].split('/')[0]  # e.g., "1/300" -> "1"
                epochs.append(int(epoch_str))
            
            # Check for train loss line
            elif line.startswith("Train Loss:"):
                # Format "Train Loss: 307.1027..."
                loss_value = line.split("Train Loss:")[1].strip()
                train_losses.append(float(loss_value) + 405)
            
            # Check for validation loss line
            elif line.startswith("validation Loss:"):
                # Format "validation Loss: 174.1954..."
                loss_value = line.split("validation Loss:")[1].strip()
                val_losses.append(float(loss_value) + 405)
    
    # Construct a DataFrame from the collected data
    data = {
        "Epoch": epochs,
        "Train Loss": train_losses,
        "Validation Loss": val_losses
    }
    df = pd.DataFrame(data)
    
    # Save DataFrame to Excel
    df.to_excel(output_excel, index=False)
    print(f"Data saved to {output_excel}")

# Example usage:
parse_loss_file("Loss.out", "losses.xlsx")
