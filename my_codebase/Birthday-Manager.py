
from utils import calculate_total_with_tax

DEFAULT_CURRENCY = "USD" 

def format_report_summary(item_prices):
    """Formats a summary line for a report."""
    total = calculate_total_with_tax(item_prices) 
    summary = f"Report Total: ${total:.2f}" 
    return summary