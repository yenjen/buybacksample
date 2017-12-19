import numpy as np
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.classifiers.morningstar import Sector
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.data import morningstar as mstar
from quantopian.pipeline.factors import (
    AverageDollarVolume,
    CustomFactor
)
from quantopian.pipeline.filters.morningstar import IsPrimaryShare

from quantopian.pipeline.factors.eventvestor import (
    BusinessDaysUntilNextEarnings,
    BusinessDaysSincePreviousEarnings,
    BusinessDaysSinceBuybackAuth,
)

from quantopian.pipeline.data.eventvestor import BuybackAuthorizations

# Constants that need to be global
COMMON_STOCK= 'ST00000001'

SECTOR_NAMES = {
 101: 'Basic Materials',
 102: 'Consumer Cyclical',
 103: 'Financial Services',
 104: 'Real Estate',
 205: 'Consumer Defensive',
 206: 'Healthcare',
 207: 'Utilities',
 308: 'Communication Services',
 309: 'Energy',
 310: 'Industrials',
 311: 'Technology' ,
}
        
# Average Dollar Volume without nanmean, so that recent IPOs are truly removed
class ADV_adj(CustomFactor):
    inputs = [USEquityPricing.close, USEquityPricing.volume]
    window_length = 252
    
    def compute(self, today, assets, out, close, volume):
        close[np.isnan(close)] = 0
        out[:] = np.mean(close * volume, 0)

def make_pipeline():
    """
    Create and return our pipeline.

    We break this piece of logic out into its own function to make it easier to
    test and modify in isolation.

    In particular, this function can be copy/pasted into research and run
    by itself.
    """
    # When current day is 6 days from a buyback announcement
    # and 10 days from an earnings
    till_earnings = BusinessDaysUntilNextEarnings()
    since_buybacks = BusinessDaysSinceBuybackAuth()
    since_earnings = BusinessDaysSincePreviousEarnings()

    # Strategy 1A: When there is a buyback announcement with a
    # known future earnings date at least 1 day ahead, go long
    # on the security starting on the buyback announcement date
    # for 25 days starting on day t-15
    # |-------buybacks & earnings-------| (earnings_announcement)
    longs = ((since_buybacks + till_earnings) <= 15) & \
            since_buybacks.notnan() & till_earnings.notnan()

    return Pipeline(
        columns={
            'till_earnings': till_earnings,
            'since_buybacks': since_buybacks,
            'since_earnings': since_earnings,
            'market_cap': mstar.valuation.market_cap.latest,
            'buyback_unit': BuybackAuthorizations.previous_unit.latest,
            'buyback_amount': BuybackAuthorizations.previous_amount.latest,
            'pricing': USEquityPricing.close.latest,
            'longs': longs
        },
        screen=(longs & universe_filters())
    )


def initialize(context):
    # Hold for a period of [-10, +15)
    context.days_to_hold = 25

    # Declares which stocks we currently held
    # and how many days we've held them dict[stock:days_held]
    context.stocks_held = {}

    # Make our pipeline
    attach_pipeline(make_pipeline(), 'buybacks_and_earnings')

    # Order our positions
    schedule_function(func=order_positions,
                      date_rule=date_rules.every_day(),
                      time_rule=time_rules.market_open())

    context.longs = None


def before_trading_start(context, data):
    results = pipeline_output('buybacks_and_earnings')
    log.info("Results: %r" % results)
    if len(results.index) == 0:
        return

    # Only look at buybacks > 5%
    results = results.apply(lambda row: convert_units(row), axis=1)
    results = results[results['Percent of SO'] > .05]
    assets_in_universe = results.index
    context.longs = assets_in_universe[results.longs]

    for stock in context.longs:
        log.info("\n")
        log.info("%s: %s days since buyback and %s left till earnings"
                 % (stock.symbol, results.ix[stock]['since_buybacks'],
                    results.ix[stock]['till_earnings']))


def convert_units(row):
    buyback_unit = row['buyback_unit']
    market_cap = row['market_cap']
    shares_outstanding = market_cap/row['pricing']
    if buyback_unit == '$M':
        total_bought = row['buyback_amount'] * 1000000.0
        percent_bought = (total_bought)/market_cap
    elif buyback_unit == "Mshares":
        percent_bought = row['buyback_amount']/shares_outstanding
    elif buyback_unit == '%':
        percent_bought = row['buyback_amount']/100.0
    else:
        percent_bought = None

    row['Percent of SO'] = percent_bought
    return row

def order_positions(context, data):
    """
    Main ordering conditions to always order an equal percentage in each
    position so it does a rolling rebalance by looking at the stocks to
    order today and the stocks we currently hold in our portfolio.
    """
    port = context.portfolio.positions
    record(leverage=context.account.leverage,
           positions=len(context.portfolio.positions))

    # Check if we've exited our positions and if we haven't, exit the
    # remaining securities that we have left
    for security in port:
        if data.can_trade(security):
            if context.stocks_held.get(security) is not None:
                context.stocks_held[security] += 1
                if context.stocks_held[security] >= context.days_to_hold:
                    order_target_percent(security, 0)
                    del context.stocks_held[security]
            # If we've deleted it but it still hasn't been exited.
            # Try exiting again
            else:
                log.info("Haven't yet exited %s, ordering again" %
                         security.symbol)
                order_target_percent(security, 0)

    if context.longs is None:
        return

    # Check our current positions
    current_longs = [pos for pos in port if
                     (port[pos].amount > 0 and pos in context.stocks_held)]
    all_longs = context.longs.tolist() + current_longs

    # Rebalance our long securities (existing + new)
    for security in all_longs:
        can_trade = context.stocks_held.get(security) <= context.days_to_hold or \
            context.stocks_held.get(security) is None
        if data.can_trade(security) and can_trade:
            order_target_percent(security, 1.0 / len(all_longs))
            if context.stocks_held.get(security) is None:
                context.stocks_held[security] = 0


def universe_filters():
    """
    Create a Pipeline producing Filters implementing common acceptance criteria.
    
    Returns
    -------
    zipline.Filter
        Filter to control tradeablility
    """

    # Equities with an average daily volume greater than 750000.
    high_volume = (AverageDollarVolume(window_length=252) > 750000)
    
    # Not Misc. sector:
    sector_check = Sector().notnull()
    
    # Equities that morningstar lists as primary shares.
    # NOTE: This will return False for stocks not in the morningstar database.
    primary_share = IsPrimaryShare()
    
    # Equities for which morningstar's most recent Market Cap value is above $300m.
    have_market_cap = mstar.valuation.market_cap.latest > 300000000
    
    # Equities not listed as depositary receipts by morningstar.
    # Note the inversion operator, `~`, at the start of the expression.
    not_depositary = ~mstar.share_class_reference.is_depositary_receipt.latest
    
    # Equities that listed as common stock (as opposed to, say, preferred stock).
    # This is our first string column. The .eq method used here produces a Filter returning
    # True for all asset/date pairs where security_type produced a value of 'ST00000001'.
    common_stock = mstar.share_class_reference.security_type.latest.eq(COMMON_STOCK)
    
    # Equities whose exchange id does not start with OTC (Over The Counter).
    # startswith() is a new method available only on string-dtype Classifiers.
    # It returns a Filter.
    not_otc = ~mstar.share_class_reference.exchange_id.latest.startswith('OTC')
    
    # Equities whose symbol (according to morningstar) ends with .WI
    # This generally indicates a "When Issued" offering.
    # endswith() works similarly to startswith().
    not_wi = ~mstar.share_class_reference.symbol.latest.endswith('.WI')
    
    # Equities whose company name ends with 'LP' or a similar string.
    # The .matches() method uses the standard library `re` module to match
    # against a regular expression.
    not_lp_name = ~mstar.company_reference.standard_name.latest.matches('.* L[\\. ]?P\.?$')
    
    # Equities with a null entry for the balance_sheet.limited_partnership field.
    # This is an alternative way of checking for LPs.
    not_lp_balance_sheet = mstar.balance_sheet.limited_partnership.latest.isnull()
    
    # Highly liquid assets only. Also eliminates IPOs in the past 12 months
    # Use new average dollar volume so that unrecorded days are given value 0
    # and not skipped over
    # S&P Criterion
    liquid = ADV_adj() > 250000
    
    # Add logic when global markets supported
    # S&P Criterion
    domicile = True
    
    # Keep it to liquid securities
    ranked_liquid = ADV_adj().rank(ascending=False) < 2000
    
    universe_filter = (high_volume & primary_share & have_market_cap & not_depositary &
                      common_stock & not_otc & not_wi & not_lp_name & not_lp_balance_sheet &
                    liquid & domicile & sector_check & liquid & ranked_liquid)
    
    return universe_filter
