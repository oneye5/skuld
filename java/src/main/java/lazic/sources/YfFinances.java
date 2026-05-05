package lazic.sources;

import java.time.LocalDate;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.time.format.DateTimeParseException;
import java.util.HashSet;
import java.util.Set;
import java.util.function.Function;

import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;

import lazic.sources.config.Tickers;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;

public class YfFinances extends DataSourceBase {

    private static final DateTimeFormatter DATE_FMT = DateTimeFormatter.ofPattern("yyyy-MM-dd");

    private final Function<String, String> fetcher;

    /** Production constructor — uses real HTTP. */
    public YfFinances() {
        this(WebHtmlGetter::get);
    }

    /** Package-private constructor for tests — inject a mock fetcher. */
    YfFinances(Function<String, String> fetcher) {
        super();
        this.fetcher = fetcher;
    }

    @Override
    public String getSourceName() { return "yf_finances"; }

    @Override
    public Set<DataPoint> getDataPoints() {
        Set<DataPoint> points = new HashSet<>();
        for (String ticker : Tickers.TICKERS) {
            try {
                String json = fetcher.apply(buildUrl(ticker));
                points.addAll(parseTimeSeries(json));
            } catch (Exception e) {
                System.err.println("YfFinances: error for " + ticker + " — " + e.getMessage());
            }
        }
        return points;
    }

    // ── Package-private for tests ────────────────────────────────────────

    static String buildUrl(String ticker) {
        return BASE_URL.replace("{TICKER}", ticker);
    }

    /**
     * Parses a Yahoo Finance fundamentals-timeseries JSON response.
     * Returns an empty set (never throws) on any malformed or null input.
     */
    static Set<DataPoint> parseTimeSeries(String json) {
        Set<DataPoint> points = new HashSet<>();
        if (json == null || json.isBlank()) return points;

        JsonObject root;
        try {
            root = new Gson().fromJson(json, JsonObject.class);
        } catch (Exception e) {
            System.err.println("YfFinances: failed to parse JSON — " + e.getMessage());
            return points;
        }

        if (!root.has("timeseries")) return points;
        JsonObject timeseries = root.getAsJsonObject("timeseries");
        if (!timeseries.has("result") || timeseries.get("result").isJsonNull()) return points;
        JsonArray results = timeseries.getAsJsonArray("result");

        for (JsonElement resultElement : results) {
            if (resultElement == null || resultElement.isJsonNull()) continue;
            JsonObject resultObj = resultElement.getAsJsonObject();

            if (!resultObj.has("meta")) continue;
            JsonObject meta = resultObj.getAsJsonObject("meta");

            JsonArray typeArr   = meta.has("type")   ? meta.getAsJsonArray("type")   : null;
            JsonArray symbolArr = meta.has("symbol") ? meta.getAsJsonArray("symbol") : null;

            if (typeArr == null || typeArr.size() == 0) continue;
            if (symbolArr == null || symbolArr.size() == 0) continue;

            String featureType = typeArr.get(0).getAsString();
            String symbol      = symbolArr.get(0).getAsString();

            if (!resultObj.has(featureType) || resultObj.get(featureType).isJsonNull()) continue;
            JsonArray dataArray = resultObj.getAsJsonArray(featureType);

            for (JsonElement elem : dataArray) {
                if (elem == null || elem.isJsonNull()) continue;
                JsonObject dataObj = elem.getAsJsonObject();

                if (!dataObj.has("asOfDate")) continue;

                LocalDateTime date;
                try {
                    date = LocalDate.parse(dataObj.get("asOfDate").getAsString(), DATE_FMT).atStartOfDay();
                } catch (DateTimeParseException e) {
                    System.err.println("YfFinances: bad date in entry — " + dataObj);
                    continue;
                }

                if (!dataObj.has("reportedValue") || dataObj.get("reportedValue").isJsonNull()) continue;
                JsonObject reportedValue = dataObj.getAsJsonObject("reportedValue");
                if (!reportedValue.has("raw")) continue;

                points.add(new DataPoint(date, symbol, featureType, reportedValue.get("raw").getAsDouble()));
            }
        }

        return points;
    }

    // ── URL ─────────────────────────────────────────────────────────────

    private static final String BASE_URL =
        "https://query1.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{TICKER}"
        + "?merge=false"
        + "&padTimeSeries=true"
        + "&period1=493590046"
        + "&period2=2750557599"
        + "&type="
        // ── Balance Sheet ──────────────────────────────────────────────
        + "annualTotalAssets,"
        + "annualTotalLiabilitiesNetMinorityInterest,"
        + "annualStockholdersEquity,"
        + "annualCashAndCashEquivalents,"
        + "annualTotalDebt,"
        + "annualNetPPE,"
        + "annualCurrentAssets,"
        + "annualCurrentLiabilities,"
        + "annualInventory,"
        + "annualAccountsReceivable,"
        + "annualRetainedEarnings,"
        + "annualCommonStock,"
        + "annualTotalEquityGrossMinorityInterest,"
        + "annualGoodwill,"
        // ── Cash Flow ─────────────────────────────────────────────────
        + "annualCashFlowsfromusedinOperatingActivitiesDirect,"
        + "annualCapitalExpenditure,"
        + "annualFreeCashFlow,"
        + "annualCashFlowFromContinuingInvestingActivities,"
        + "annualCashFlowFromContinuingFinancingActivities,"
        + "annualCashDividendsPaid,"
        + "annualIssuanceOfDebt,"
        + "annualRepaymentOfDebt,"
        // ── Income Statement (existing) ────────────────────────────────
        + "annualTaxEffectOfUnusualItems,trailingTaxEffectOfUnusualItems,"
        + "annualTaxRateForCalcs,trailingTaxRateForCalcs,"
        + "annualNormalizedEBITDA,trailingNormalizedEBITDA,"
        + "annualNormalizedDilutedEPS,trailingNormalizedDilutedEPS,"
        + "annualNormalizedBasicEPS,trailingNormalizedBasicEPS,"
        + "annualTotalUnusualItems,trailingTotalUnusualItems,"
        + "annualTotalUnusualItemsExcludingGoodwill,trailingTotalUnusualItemsExcludingGoodwill,"
        + "annualNetIncomeFromContinuingOperationNetMinorityInterest,trailingNetIncomeFromContinuingOperationNetMinorityInterest,"
        + "annualReconciledDepreciation,trailingReconciledDepreciation,"
        + "annualEBITDA,trailingEBITDA,"
        + "annualEBIT,trailingEBIT,"
        + "annualTotalMoneyMarketInvestments,trailingTotalMoneyMarketInvestments,"
        + "annualContinuingAndDiscontinuedDilutedEPS,trailingContinuingAndDiscontinuedDilutedEPS,"
        + "annualContinuingAndDiscontinuedBasicEPS,trailingContinuingAndDiscontinuedBasicEPS,"
        + "annualNormalizedIncome,trailingNormalizedIncome,"
        + "annualNetIncomeFromContinuingAndDiscontinuedOperation,trailingNetIncomeFromContinuingAndDiscontinuedOperation,"
        + "annualInterestIncomeAfterProvisionForLoanLoss,trailingInterestIncomeAfterProvisionForLoanLoss,"
        + "annualRentExpenseSupplemental,trailingRentExpenseSupplemental,"
        + "annualReportedNormalizedDilutedEPS,trailingReportedNormalizedDilutedEPS,"
        + "annualReportedNormalizedBasicEPS,trailingReportedNormalizedBasicEPS,"
        + "annualDividendPerShare,trailingDividendPerShare,"
        + "annualDilutedAverageShares,trailingDilutedAverageShares,"
        + "annualBasicAverageShares,trailingBasicAverageShares,"
        + "annualDilutedEPS,trailingDilutedEPS,"
        + "annualDilutedEPSOtherGainsLosses,trailingDilutedEPSOtherGainsLosses,"
        + "annualTaxLossCarryforwardDilutedEPS,trailingTaxLossCarryforwardDilutedEPS,"
        + "annualDilutedAccountingChange,trailingDilutedAccountingChange,"
        + "annualDilutedExtraordinary,trailingDilutedExtraordinary,"
        + "annualDilutedDiscontinuousOperations,trailingDilutedDiscontinuousOperations,"
        + "annualDilutedContinuousOperations,trailingDilutedContinuousOperations,"
        + "annualBasicEPS,trailingBasicEPS,"
        + "annualBasicEPSOtherGainsLosses,trailingBasicEPSOtherGainsLosses,"
        + "annualTaxLossCarryforwardBasicEPS,trailingTaxLossCarryforwardBasicEPS,"
        + "annualBasicAccountingChange,trailingBasicAccountingChange,"
        + "annualBasicExtraordinary,trailingBasicExtraordinary,"
        + "annualBasicDiscontinuousOperations,trailingBasicDiscontinuousOperations,"
        + "annualBasicContinuousOperations,trailingBasicContinuousOperations,"
        + "annualDilutedNIAvailtoComStockholders,trailingDilutedNIAvailtoComStockholders,"
        + "annualAverageDilutionEarnings,trailingAverageDilutionEarnings,"
        + "annualNetIncomeCommonStockholders,trailingNetIncomeCommonStockholders,"
        + "annualOtherThanPreferredStockDividend,trailingOtherThanPreferredStockDividend,"
        + "annualPreferredStockDividends,trailingPreferredStockDividends,"
        + "annualNetIncome,trailingNetIncome,"
        + "annualMinorityInterests,trailingMinorityInterests,"
        + "annualNetIncomeIncludingNoncontrollingInterests,trailingNetIncomeIncludingNoncontrollingInterests,"
        + "annualNetIncomeFromTaxLossCarryforward,trailingNetIncomeFromTaxLossCarryforward,"
        + "annualNetIncomeExtraordinary,trailingNetIncomeExtraordinary,"
        + "annualNetIncomeDiscontinuousOperations,trailingNetIncomeDiscontinuousOperations,"
        + "annualNetIncomeContinuousOperations,trailingNetIncomeContinuousOperations,"
        + "annualEarningsFromEquityInterestNetOfTax,trailingEarningsFromEquityInterestNetOfTax,"
        + "annualTaxProvision,trailingTaxProvision,"
        + "annualPretaxIncome,trailingPretatIncome,"
        + "annualOtherNonOperatingIncomeExpenses,trailingOtherNonOperatingIncomeExpenses,"
        + "annualSpecialIncomeCharges,trailingSpecialIncomeCharges,"
        + "annualOtherSpecialCharges,trailingOtherSpecialCharges,"
        + "annualLossonExtinguishmentofDebt,trailingLossonExtinguishmentofDebt,"
        + "annualWriteOff,trailingWriteOff,"
        + "annualImpairmentOfCapitalAssets,trailingImpairmentOfCapitalAssets,"
        + "annualRestructuringAndMergernAcquisition,trailingRestructuringAndMergernAcquisition,"
        + "annualGainOnSaleOfBusiness,trailingGainOnSaleOfBusiness,"
        + "annualIncomefromAssociatesandOtherParticipatingInterests,"
        + "trailingIncomefromAssociatesandOtherParticipatingInterests,"
        + "annualNonInterestExpense,trailingNonInterestExpense,"
        + "annualOtherNonInterestExpense,trailingOtherNonInterestExpense,"
        + "annualSecuritiesAmortization,trailingSecuritiesAmortization,"
        + "annualDepreciationAmortizationDepletionIncomeStatement,trailingDepreciationAmortizationDepletionIncomeStatement,"
        + "annualDepletionIncomeStatement,trailingDepletionIncomeStatement,"
        + "annualDepreciationAndAmortizationInIncomeStatement,trailingDepreciationAndAmortizationInIncomeStatement,"
        + "annualAmortization,trailingAmortization,"
        + "annualAmortizationOfIntangiblesIncomeStatement,trailingAmortizationOfIntangiblesIncomeStatement,"
        + "annualDepreciationIncomeStatement,trailingDepreciationIncomeStatement,"
        + "annualSellingGeneralAndAdministration,trailingSellingGeneralAndAdministration,"
        + "annualSellingAndMarketingExpense,trailingSellingAndMarketingExpense,"
        + "annualGeneralAndAdministrativeExpense,trailingGeneralAndAdministrativeExpense,"
        + "annualOtherGandA,trailingOtherGandA,"
        + "annualInsuranceAndClaims,trailingInsuranceAndClaims,"
        + "annualRentAndLandingFees,trailingRentAndLandingFees,"
        + "annualSalariesAndWages,trailingSalariesAndWages,"
        + "annualProfessionalExpenseAndContractServicesExpense,trailingProfessionalExpenseAndContractServicesExpense,"
        + "annualOccupancyAndEquipment,trailingOccupancyAndEquipment,"
        + "annualEquipment,trailingEquipment,"
        + "annualNetOccupancyExpense,trailingNetOccupancyExpense,"
        + "annualCreditLossesProvision,trailingCreditLossesProvision,"
        + "annualTotalRevenue,trailingTotalRevenue,"
        + "annualNonInterestIncome,trailingNonInterestIncome,"
        + "annualOtherNonInterestIncome,trailingOtherNonInterestIncome,"
        + "annualGainLossonSaleofAssets,trailingGainLossonSaleofAssets,"
        + "annualGainonSaleofInvestmentProperty,trailingGainonSaleofInvestmentProperty,"
        + "annualGainonSaleofLoans,trailingGainonSaleofLoans,"
        + "annualGainOnSaleOfSecurity,trailingGainOnSaleOfSecurity,"
        + "annualForeignExchangeTradingGains,trailingForeignExchangeTradingGains,"
        + "annualTradingGainLoss,trailingTradingGainLoss,"
        + "annualInvestmentBankingProfit,trailingInvestmentBankingProfit,"
        + "annualDividendIncome,trailingDividendIncome,"
        + "annualFeesAndCommissions,trailingFeesAndCommissions,"
        + "annualFeesandCommissionExpense,trailingFeesandCommissionExpense,"
        + "annualFeesandCommissionIncome,trailingFeesandCommissionIncome,"
        + "annualOtherCustomerServices,trailingOtherCustomerServices,"
        + "annualCreditCard,trailingCreditCard,"
        + "annualSecuritiesActivities,trailingSecuritiesActivities,"
        + "annualTrustFeesbyCommissions,trailingTrustFeesbyCommissions,"
        + "annualServiceChargeOnDepositorAccounts,trailingServiceChargeOnDepositorAccounts,"
        + "annualTotalPremiumsEarned,trailingTotalPremiumsEarned,"
        + "annualNetInterestIncome,trailingNetInterestIncome,"
        + "annualInterestExpense,trailingInterestExpense,"
        + "annualOtherInterestExpense,trailingOtherInterestExpense,"
        + "annualInterestExpenseForFederalFundsSoldAndSecuritiesPurchaseUnderAgreementsToResell,"
        + "trailingInterestExpenseForFederalFundsSoldAndSecuritiesPurchaseUnderAgreementsToResell,"
        + "annualInterestExpenseForLongTermDebtAndCapitalSecurities,"
        + "trailingInterestExpenseForLongTermDebtAndCapitalSecurities,"
        + "annualInterestExpenseForShortTermDebt,trailingInterestExpenseForShortTermDebt,"
        + "annualInterestExpenseForDeposit,trailingInterestExpenseForDeposit,"
        + "annualInterestIncome,trailingInterestIncome,"
        + "annualOtherInterestIncome,trailingOtherInterestIncome,"
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
