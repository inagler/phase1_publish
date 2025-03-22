import sys

def main():
    if len(sys.argv) != 4:
        print(f"Usage: {sys.argv[0]} <param1> <param2> <param3>")
        sys.exit(1)
    
    param1 = sys.argv[1]
    param2 = sys.argv[2]
    param3 = sys.argv[3]

    print(f"Called with parameters: {param1}, {param2}, {param3}")

if __name__ == "__main__":
    main()