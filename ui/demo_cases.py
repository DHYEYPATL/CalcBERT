# ui/demo_cases.py

DEMO_CASES = [
    "STARBCKS #103 MUMBAI 12:32PM",
    "MCDONLDS 456 BANGALORE",
    "HP PUMP PETROL 3456",
    "PAYTM UPI - REFILL",
    "ICICI BANK NEFT 9876543210",
    "AMAZON Mktplace CA 4343",
    "FLIPKART ORDER ID 12345",
    "BIGBASKET GROCERIES 4321"
]

# Optional: mapping of index -> correction label to simulate user corrections in demo
DEMO_CORRECTIONS = {
    # case index: correct label
    2: "Fuel",               # HP PUMP PETROL => Fuel
    5: "Shopping",           # Amazon marketplace
}
