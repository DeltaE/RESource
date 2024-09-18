import argparse

def create_resources(resource_type):
    if resource_type == 'solar':
        print("Running script for solar resources")
        # Add your logic for solar resources here
    elif resource_type == 'bess':
        print("Running script for BESS resources")
        # Add your logic for BESS resources here
    elif resource_type == 'wind':
        print("Running script for wind resources")
        # Add your logic for wind resources here
    else:
        print("Unknown resource type")

def main():
    # Set up argument parsing
    parser = argparse.ArgumentParser(description='Linking Tool CLI')
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest='command')
    
    # Subparser for 'create_resources' command
    create_resources_parser = subparsers.add_parser('create_resources', help='Create resources')
    create_resources_parser.add_argument('resource_type', choices=['solar', 'bess', 'wind'], help='Type of resource to create')
    
    # Parse the arguments
    args = parser.parse_args()

    # Run the corresponding function based on the command
    if args.command == 'create_resources':
        create_resources(args.resource_type)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
