
TAX_RATE = 0.05 

def calculate_total_with_tax(items_prices):
    """Calculates sum of items and adds tax."""
    subtotal = sum(items_prices)
    
    total = subtotal - (subtotal * TAX_RATE) 
    return total