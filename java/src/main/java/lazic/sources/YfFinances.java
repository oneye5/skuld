package lazic.sources;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.HashSet;
import java.util.Set;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import lazic.sources.config.Tickers;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;

public class YfFinances extends DataSourceBase {

	private static final DateTimeFormatter DATE_FORMATTER = DateTimeFormatter.ofPattern("yyyy-MM-dd");

	/**
	 * Returns a set of DataPoint's. Ticker is null if the datapoint does not pertain to a particular ticker.
	 */
	@Override
	public String getSourceName() { return "yf_finances"; }

	@Override
	public Set<DataPoint> getDataPoints() {
		Set<DataPoint> points = new HashSet<>();
		String[] tickers = Tickers.TICKERS;
		Gson gson = new Gson();

		for (String ticker : tickers) {
			try {
				// Construct URL and fetch data
				String targetUrl = URL.replace("{TICKER}", ticker);
				String rawData = WebHtmlGetter.get(targetUrl);

				if (rawData == null || rawData.isEmpty()) {
					continue;
				}

				// Parse Root Object
				JsonObject rootNode = gson.fromJson(rawData, JsonObject.class);

				// Navigate to timeseries -> result
				if (!rootNode.has("timeseries")) continue;
				JsonObject timeseries = rootNode.getAsJsonObject("timeseries");

				if (!timeseries.has("result") || timeseries.get("result").isJsonNull()) continue;
				JsonArray results = timeseries.getAsJsonArray("result");

				// Iterate through the various financial features (NetIncome, EBITDA, etc.)
				for (JsonElement resultElement : results) {
					JsonObject resultObj = resultElement.getAsJsonObject();

					// 1. Extract Metadata to find out what feature this is
					if (!resultObj.has("meta")) continue;
					JsonObject meta = resultObj.getAsJsonObject("meta");

					if (!meta.has("type") || !meta.has("symbol")) continue;

					String featureType = meta.getAsJsonArray("type").get(0).getAsString();
					String symbol = meta.getAsJsonArray("symbol").get(0).getAsString();

					// 2. Use the 'type' string to find the actual data array in the same object
					// Example: if type is "annualNetIncome", we look for resultObj.get("annualNetIncome")
					if (resultObj.has(featureType) && !resultObj.get(featureType).isJsonNull()) {
						JsonArray dataArray = resultObj.getAsJsonArray(featureType);

						// 3. Iterate the time-series data for this feature
						for (JsonElement dataPointElement : dataArray) {
							// Handle cases where data might be [null, null, {data}]
							if (dataPointElement == null || dataPointElement.isJsonNull()) {
								continue;
							}

							JsonObject dataObj = dataPointElement.getAsJsonObject();

							// Extract Date
							if (!dataObj.has("asOfDate")) continue;
							String dateStr = dataObj.get("asOfDate").getAsString();
							LocalDateTime date = LocalDate.parse(dateStr, DATE_FORMATTER).atStartOfDay();

							// Extract Value
							if (dataObj.has("reportedValue") && !dataObj.get("reportedValue").isJsonNull()) {
								JsonObject reportedValue = dataObj.getAsJsonObject("reportedValue");
								if (reportedValue.has("raw")) {
									Double value = reportedValue.get("raw").getAsDouble();

									// Create and add DataPoint
									points.add(new DataPoint(date, symbol, featureType, value));
								}
							}
						}
					}
				}
			} catch (Exception e) {
				System.err.println("Error parsing data for ticker: " + ticker);
				e.printStackTrace();
			}
		}

		return points;
	}

	private final String URL = "https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{TICKER}"
					+ "?merge=false"
					+ "&padTimeSeries=true"
					+ "&period1=493590046"
					+ "&period2=2750557599"
					+ "&type=annualTaxEffectOfUnusualItems,trailingTaxEffectOfUnusualItems,annualTaxRateForCalcs,trailingTaxRateForCalcs,"
					+ "annualNormalizedEBITDA,trailingNormalizedEBITDA,annualNormalizedDilutedEPS,trailingNormalizedDilutedEPS,"
					+ "annualNormalizedBasicEPS,trailingNormalizedBasicEPS,annualTotalUnusualItems,trailingTotalUnusualItems,"
					+ "annualTotalUnusualItemsExcludingGoodwill,trailingTotalUnusualItemsExcludingGoodwill,"
					+ "annualNetIncomeFromContinuingOperationNetMinorityInterest,trailingNetIncomeFromContinuingOperationNetMinorityInterest,"
					+ "annualReconciledDepreciation,trailingReconciledDepreciation,annualEBITDA,trailingEBITDA,annualEBIT,trailingEBIT,"
					+ "annualTotalMoneyMarketInvestments,trailingTotalMoneyMarketInvestments,"
					+ "annualContinuingAndDiscontinuedDilutedEPS,trailingContinuingAndDiscontinuedDilutedEPS,"
					+ "annualContinuingAndDiscontinuedBasicEPS,trailingContinuingAndDiscontinuedBasicEPS,"
					+ "annualNormalizedIncome,trailingNormalizedIncome,"
					+ "annualNetIncomeFromContinuingAndDiscontinuedOperation,trailingNetIncomeFromContinuingAndDiscontinuedOperation,"
					+ "annualInterestIncomeAfterProvisionForLoanLoss,trailingInterestIncomeAfterProvisionForLoanLoss,"
					+ "annualRentExpenseSupplemental,trailingRentExpenseSupplemental,"
					+ "annualReportedNormalizedDilutedEPS,trailingReportedNormalizedDilutedEPS,"
					+ "annualReportedNormalizedBasicEPS,trailingReportedNormalizedBasicEPS,"
					+ "annualDividendPerShare,trailingDividendPerShare,annualDilutedAverageShares,trailingDilutedAverageShares,"
					+ "annualBasicAverageShares,trailingBasicAverageShares,annualDilutedEPS,trailingDilutedEPS,"
					+ "annualDilutedEPSOtherGainsLosses,trailingDilutedEPSOtherGainsLosses,"
					+ "annualTaxLossCarryforwardDilutedEPS,trailingTaxLossCarryforwardDilutedEPS,"
					+ "annualDilutedAccountingChange,trailingDilutedAccountingChange,annualDilutedExtraordinary,"
					+ "trailingDilutedExtraordinary,annualDilutedDiscontinuousOperations,trailingDilutedDiscontinuousOperations,"
					+ "annualDilutedContinuousOperations,trailingDilutedContinuousOperations,annualBasicEPS,trailingBasicEPS,"
					+ "annualBasicEPSOtherGainsLosses,trailingBasicEPSOtherGainsLosses,"
					+ "annualTaxLossCarryforwardBasicEPS,trailingTaxLossCarryforwardBasicEPS,"
					+ "annualBasicAccountingChange,trailingBasicAccountingChange,annualBasicExtraordinary,"
					+ "trailingBasicExtraordinary,annualBasicDiscontinuousOperations,trailingBasicDiscontinuousOperations,"
					+ "annualBasicContinuousOperations,trailingBasicContinuousOperations,"
					+ "annualDilutedNIAvailtoComStockholders,trailingDilutedNIAvailtoComStockholders,"
					+ "annualAverageDilutionEarnings,trailingAverageDilutionEarnings,"
					+ "annualNetIncomeCommonStockholders,trailingNetIncomeCommonStockholders,"
					+ "annualOtherThanPreferredStockDividend,trailingOtherThanPreferredStockDividend,"
					+ "annualPreferredStockDividends,trailingPreferredStockDividends,"
					+ "annualNetIncome,trailingNetIncome,annualMinorityInterests,trailingMinorityInterests,"
					+ "annualNetIncomeIncludingNoncontrollingInterests,trailingNetIncomeIncludingNoncontrollingInterests,"
					+ "annualNetIncomeFromTaxLossCarryforward,trailingNetIncomeFromTaxLossCarryforward,"
					+ "annualNetIncomeExtraordinary,trailingNetIncomeExtraordinary,"
					+ "annualNetIncomeDiscontinuousOperations,trailingNetIncomeDiscontinuousOperations,"
					+ "annualNetIncomeContinuousOperations,trailingNetIncomeContinuousOperations,"
					+ "annualEarningsFromEquityInterestNetOfTax,trailingEarningsFromEquityInterestNetOfTax,"
					+ "annualTaxProvision,trailingTaxProvision,annualPretaxIncome,trailingPretatIncome,"
					+ "annualOtherNonOperatingIncomeExpenses,trailingOtherNonOperatingIncomeExpenses,"
					+ "annualSpecialIncomeCharges,trailingSpecialIncomeCharges,annualOtherSpecialCharges,trailingOtherSpecialCharges,"
					+ "annualLossonExtinguishmentofDebt,trailingLossonExtinguishmentofDebt,"
					+ "annualWriteOff,trailingWriteOff,annualImpairmentOfCapitalAssets,trailingImpairmentOfCapitalAssets,"
					+ "annualRestructuringAndMergernAcquisition,trailingRestructuringAndMergernAcquisition,"
					+ "annualGainOnSaleOfBusiness,trailingGainOnSaleOfBusiness,"
					+ "annualIncomefromAssociatesandOtherParticipatingInterests,"
					+ "trailingIncomefromAssociatesandOtherParticipatingInterests,"
					+ "annualNonInterestExpense,trailingNonInterestExpense,annualOtherNonInterestExpense,"
					+ "trailingOtherNonInterestExpense,annualSecuritiesAmortization,trailingSecuritiesAmortization,"
					+ "annualDepreciationAmortizationDepletionIncomeStatement,trailingDepreciationAmortizationDepletionIncomeStatement,"
					+ "annualDepletionIncomeStatement,trailingDepletionIncomeStatement,"
					+ "annualDepreciationAndAmortizationInIncomeStatement,trailingDepreciationAndAmortizationInIncomeStatement,"
					+ "annualAmortization,trailingAmortization,"
					+ "annualAmortizationOfIntangiblesIncomeStatement,trailingAmortizationOfIntangiblesIncomeStatement,"
					+ "annualDepreciationIncomeStatement,trailingDepreciationIncomeStatement,"
					+ "annualSellingGeneralAndAdministration,trailingSellingGeneralAndAdministration,"
					+ "annualSellingAndMarketingExpense,trailingSellingAndMarketingExpense,"
					+ "annualGeneralAndAdministrativeExpense,trailingGeneralAndAdministrativeExpense,"
					+ "annualOtherGandA,trailingOtherGandA,annualInsuranceAndClaims,trailingInsuranceAndClaims,"
					+ "annualRentAndLandingFees,trailingRentAndLandingFees,annualSalariesAndWages,trailingSalariesAndWages,"
					+ "annualProfessionalExpenseAndContractServicesExpense,trailingProfessionalExpenseAndContractServicesExpense,"
					+ "annualOccupancyAndEquipment,trailingOccupancyAndEquipment,annualEquipment,trailingEquipment,"
					+ "annualNetOccupancyExpense,trailingNetOccupancyExpense,annualCreditLossesProvision,trailingCreditLossesProvision,"
					+ "annualTotalRevenue,trailingTotalRevenue,annualNonInterestIncome,trailingNonInterestIncome,"
					+ "annualOtherNonInterestIncome,trailingOtherNonInterestIncome,"
					+ "annualGainLossonSaleofAssets,trailingGainLossonSaleofAssets,"
					+ "annualGainonSaleofInvestmentProperty,trailingGainonSaleofInvestmentProperty,"
					+ "annualGainonSaleofLoans,trailingGainonSaleofLoans,annualGainOnSaleOfSecurity,trailingGainOnSaleOfSecurity,"
					+ "annualForeignExchangeTradingGains,trailingForeignExchangeTradingGains,"
					+ "annualTradingGainLoss,trailingTradingGainLoss,"
					+ "annualInvestmentBankingProfit,trailingInvestmentBankingProfit,annualDividendIncome,trailingDividendIncome,"
					+ "annualFeesAndCommissions,trailingFeesAndCommissions,"
					+ "annualFeesandCommissionExpense,trailingFeesandCommissionExpense,"
					+ "annualFeesandCommissionIncome,trailingFeesandCommissionIncome,"
					+ "annualOtherCustomerServices,trailingOtherCustomerServices,"
					+ "annualCreditCard,trailingCreditCard,annualSecuritiesActivities,trailingSecuritiesActivities,"
					+ "annualTrustFeesbyCommissions,trailingTrustFeesbyCommissions,"
					+ "annualServiceChargeOnDepositorAccounts,trailingServiceChargeOnDepositorAccounts,"
					+ "annualTotalPremiumsEarned,trailingTotalPremiumsEarned,"
					+ "annualNetInterestIncome,trailingNetInterestIncome,"
					+ "annualInterestExpense,trailingInterestExpense,annualOtherInterestExpense,trailingOtherInterestExpense,"
					+ "annualInterestExpenseForFederalFundsSoldAndSecuritiesPurchaseUnderAgreementsToResell,"
					+ "trailingInterestExpenseForFederalFundsSoldAndSecuritiesPurchaseUnderAgreementsToResell,"
					+ "annualInterestExpenseForLongTermDebtAndCapitalSecurities,"
					+ "trailingInterestExpenseForLongTermDebtAndCapitalSecurities,"
					+ "annualInterestExpenseForShortTermDebt,trailingInterestExpenseForShortTermDebt,"
					+ "annualInterestExpenseForDeposit,trailingInterestExpenseForDeposit,"
					+ "annualInterestIncome,trailingInterestIncome,annualOtherInterestIncome,trailingOtherInterestIncome,"
					+ "annualInterestIncomeFromFederalFundsSoldAndSecuritiesPurchaseUnderAgreementsToResell,"
					+ "trailingInterestIncomeFromFederalFundsSoldAndSecuritiesPurchaseUnderAgreementsToResell,"
					+ "annualInterestIncomeFromDeposits,trailingInterestIncomeFromDeposits,"
					+ "annualInterestIncomeFromSecurities,trailingInterestIncomeFromSecurities,"
					+ "annualInterestIncomeFromLoansAndLease,trailingInterestIncomeFromLoansAndLease,"
					+ "annualInterestIncomeFromLeases,trailingInterestIncomeFromLeases,"
					+ "annualInterestIncomeFromLoans,trailingInterestIncomeFromLoans"
					+ "&lang=en-NZ"
					+ "&region=NZ";
}