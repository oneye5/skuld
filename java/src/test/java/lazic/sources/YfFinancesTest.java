package lazic.sources;

import lazic.utils.ingest.DataPoint;
import org.junit.jupiter.api.Test;

import java.time.LocalDateTime;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

class YfFinancesTest {

    // ──────────────────────────────────────────────
    // Helpers — minimal valid YF timeseries JSON
    // ──────────────────────────────────────────────

    /** Builds a minimal but structurally valid timeseries JSON for one type + one data point. */
    private static String validJson(String ticker, String type, String date, double rawValue) {
        return "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[\"" + type + "\"],\"symbol\":[\"" + ticker + "\"]},"
            + "    \"" + type + "\":[{"
            + "      \"asOfDate\":\"" + date + "\","
            + "      \"reportedValue\":{\"raw\":" + rawValue + "}"
            + "    }]"
            + "  }]"
            + "}"
            + "}";
    }

    private static final String TICKER = "FPH.NZ";
    private static final String TYPE   = "annualTotalAssets";
    private static final String DATE   = "2023-03-31";
    private static final double VALUE  = 2204000000.0;

    // ──────────────────────────────────────────────
    // parseTimeSeries — null / empty / malformed
    // ──────────────────────────────────────────────

    @Test
    void parseTimeSeries_nullInput_returnsEmptySet() {
        Set<DataPoint> result = YfFinances.parseTimeSeries(null);
        assertTrue(result.isEmpty(), "null JSON must return empty set, not throw");
    }

    @Test
    void parseTimeSeries_emptyString_returnsEmptySet() {
        Set<DataPoint> result = YfFinances.parseTimeSeries("");
        assertTrue(result.isEmpty());
    }

    @Test
    void parseTimeSeries_whitespaceOnly_returnsEmptySet() {
        Set<DataPoint> result = YfFinances.parseTimeSeries("   \n");
        assertTrue(result.isEmpty());
    }

    @Test
    void parseTimeSeries_invalidJson_returnsEmptySetWithoutThrowing() {
        Set<DataPoint> result = YfFinances.parseTimeSeries("this is not json {{{}");
        assertTrue(result.isEmpty(), "Garbage input must not throw, must return empty");
    }

    @Test
    void parseTimeSeries_emptyJsonObject_returnsEmptySet() {
        assertTrue(YfFinances.parseTimeSeries("{}").isEmpty());
    }

    @Test
    void parseTimeSeries_missingTimeseriesKey_returnsEmptySet() {
        assertTrue(YfFinances.parseTimeSeries("{\"other\":{}}").isEmpty());
    }

    @Test
    void parseTimeSeries_nullResultArray_returnsEmptySet() {
        String json = "{\"timeseries\":{\"result\":null}}";
        assertTrue(YfFinances.parseTimeSeries(json).isEmpty());
    }

    @Test
    void parseTimeSeries_emptyResultArray_returnsEmptySet() {
        String json = "{\"timeseries\":{\"result\":[]}}";
        assertTrue(YfFinances.parseTimeSeries(json).isEmpty());
    }

    // ──────────────────────────────────────────────
    // parseTimeSeries — happy path
    // ──────────────────────────────────────────────

    @Test
    void parseTimeSeries_validSingleDataPoint_returnsCorrectDataPoint() {
        Set<DataPoint> result = YfFinances.parseTimeSeries(validJson(TICKER, TYPE, DATE, VALUE));

        assertEquals(1, result.size(), "Expected exactly one DataPoint");
        DataPoint dp = result.iterator().next();
        assertEquals(TICKER, dp.getTicker());
        assertEquals(TYPE,   dp.getFeatureName());
        assertEquals(VALUE,  dp.getValue(), 0.01);
        assertEquals(LocalDateTime.of(2023, 3, 31, 0, 0), dp.getTimestamp());
    }

    @Test
    void parseTimeSeries_multipleTypesInResponse_returnsOneDataPointPerType() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":["
            + "    {\"meta\":{\"type\":[\"annualTotalAssets\"],\"symbol\":[\"SPK.NZ\"]},"
            + "     \"annualTotalAssets\":[{\"asOfDate\":\"2023-06-30\",\"reportedValue\":{\"raw\":4482000000.0}}]},"
            + "    {\"meta\":{\"type\":[\"annualStockholdersEquity\"],\"symbol\":[\"SPK.NZ\"]},"
            + "     \"annualStockholdersEquity\":[{\"asOfDate\":\"2023-06-30\",\"reportedValue\":{\"raw\":1940000000.0}}]}"
            + "  ]"
            + "}"
            + "}";

        Set<DataPoint> result = YfFinances.parseTimeSeries(json);
        assertEquals(2, result.size());
    }

    @Test
    void parseTimeSeries_multipleYearsForSameType_returnsOneDataPointPerYear() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[\"annualTotalAssets\"],\"symbol\":[\"FPH.NZ\"]},"
            + "    \"annualTotalAssets\":["
            + "      {\"asOfDate\":\"2022-03-31\",\"reportedValue\":{\"raw\":2107000000.0}},"
            + "      {\"asOfDate\":\"2023-03-31\",\"reportedValue\":{\"raw\":2204000000.0}}"
            + "    ]"
            + "  }]"
            + "}"
            + "}";

        assertEquals(2, YfFinances.parseTimeSeries(json).size());
    }

    // ──────────────────────────────────────────────
    // parseTimeSeries — resilience: malformed entries
    // ──────────────────────────────────────────────

    @Test
    void parseTimeSeries_nullEntryInDataArray_skipsNullKeepingValidEntries() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[\"annualTotalAssets\"],\"symbol\":[\"FPH.NZ\"]},"
            + "    \"annualTotalAssets\":[null,null,"
            + "      {\"asOfDate\":\"2023-03-31\",\"reportedValue\":{\"raw\":2204000000.0}}"
            + "    ]"
            + "  }]"
            + "}"
            + "}";

        Set<DataPoint> result = YfFinances.parseTimeSeries(json);
        assertEquals(1, result.size(), "Null padding must be skipped; valid entry must survive");
    }

    @Test
    void parseTimeSeries_missingAsOfDate_skipsEntry() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[\"annualTotalAssets\"],\"symbol\":[\"FPH.NZ\"]},"
            + "    \"annualTotalAssets\":[{\"reportedValue\":{\"raw\":2204000000.0}}]"
            + "  }]"
            + "}"
            + "}";

        assertTrue(YfFinances.parseTimeSeries(json).isEmpty(),
            "Entry without asOfDate must be skipped");
    }

    @Test
    void parseTimeSeries_malformedDate_skipsEntryWithoutThrowing() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[\"annualTotalAssets\"],\"symbol\":[\"FPH.NZ\"]},"
            + "    \"annualTotalAssets\":[{\"asOfDate\":\"not-a-date\",\"reportedValue\":{\"raw\":1.0}}]"
            + "  }]"
            + "}"
            + "}";

        assertDoesNotThrow(() -> YfFinances.parseTimeSeries(json));
        assertTrue(YfFinances.parseTimeSeries(json).isEmpty(),
            "Entry with malformed date must be skipped");
    }

    @Test
    void parseTimeSeries_missingReportedValue_skipsEntry() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[\"annualTotalAssets\"],\"symbol\":[\"FPH.NZ\"]},"
            + "    \"annualTotalAssets\":[{\"asOfDate\":\"2023-03-31\"}]"
            + "  }]"
            + "}"
            + "}";

        assertTrue(YfFinances.parseTimeSeries(json).isEmpty());
    }

    @Test
    void parseTimeSeries_nullReportedValue_skipsEntry() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[\"annualTotalAssets\"],\"symbol\":[\"FPH.NZ\"]},"
            + "    \"annualTotalAssets\":[{\"asOfDate\":\"2023-03-31\",\"reportedValue\":null}]"
            + "  }]"
            + "}"
            + "}";

        assertTrue(YfFinances.parseTimeSeries(json).isEmpty());
    }

    @Test
    void parseTimeSeries_reportedValueMissingRawKey_skipsEntry() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[\"annualTotalAssets\"],\"symbol\":[\"FPH.NZ\"]},"
            + "    \"annualTotalAssets\":[{\"asOfDate\":\"2023-03-31\","
            + "      \"reportedValue\":{\"fmt\":\"2.2B\"}}]"
            + "  }]"
            + "}"
            + "}";

        assertTrue(YfFinances.parseTimeSeries(json).isEmpty(),
            "reportedValue with no 'raw' key must be skipped");
    }

    @Test
    void parseTimeSeries_emptyTypeArray_skipsResultElement() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[],\"symbol\":[\"FPH.NZ\"]},"
            + "    \"annualTotalAssets\":[{\"asOfDate\":\"2023-03-31\",\"reportedValue\":{\"raw\":1.0}}]"
            + "  }]"
            + "}"
            + "}";

        assertTrue(YfFinances.parseTimeSeries(json).isEmpty(),
            "Empty type array must not cause ArrayIndexOutOfBoundsException");
    }

    @Test
    void parseTimeSeries_emptySymbolArray_skipsResultElement() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[\"annualTotalAssets\"],\"symbol\":[]},"
            + "    \"annualTotalAssets\":[{\"asOfDate\":\"2023-03-31\",\"reportedValue\":{\"raw\":1.0}}]"
            + "  }]"
            + "}"
            + "}";

        assertTrue(YfFinances.parseTimeSeries(json).isEmpty(),
            "Empty symbol array must not cause ArrayIndexOutOfBoundsException");
    }

    @Test
    void parseTimeSeries_metaMissingTypeKey_skipsResultElement() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"symbol\":[\"FPH.NZ\"]},"
            + "    \"annualTotalAssets\":[{\"asOfDate\":\"2023-03-31\",\"reportedValue\":{\"raw\":1.0}}]"
            + "  }]"
            + "}"
            + "}";

        assertTrue(YfFinances.parseTimeSeries(json).isEmpty());
    }

    @Test
    void parseTimeSeries_typeNameNotPresentAsKeyInResult_skipsResultElement() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"meta\":{\"type\":[\"annualTotalAssets\"],\"symbol\":[\"FPH.NZ\"]},"
            + "    \"annualNetIncome\":[{\"asOfDate\":\"2023-03-31\",\"reportedValue\":{\"raw\":1.0}}]"
            + "  }]"
            + "}"
            + "}";

        assertTrue(YfFinances.parseTimeSeries(json).isEmpty(),
            "Mismatch between meta.type and data key must produce empty result, not NPE");
    }

    @Test
    void parseTimeSeries_missingMetaKey_skipsResultElement() {
        String json = "{"
            + "\"timeseries\":{"
            + "  \"result\":[{"
            + "    \"annualTotalAssets\":[{\"asOfDate\":\"2023-03-31\",\"reportedValue\":{\"raw\":1.0}}]"
            + "  }]"
            + "}"
            + "}";

        assertTrue(YfFinances.parseTimeSeries(json).isEmpty(),
            "Result element with no 'meta' key must be skipped, not NPE");
    }

    // ──────────────────────────────────────────────
    // buildUrl — type coverage
    // ──────────────────────────────────────────────

    @Test
    void buildUrl_containsTickerInPath() {
        String url = YfFinances.buildUrl("FPH.NZ");
        assertTrue(url.contains("FPH.NZ"), "URL must contain the ticker symbol");
    }

    @Test
    void buildUrl_containsAllRequiredBalanceSheetTypes() {
        String url = YfFinances.buildUrl("ANY.NZ");
        String[] required = {
            "annualTotalAssets",
            "annualTotalLiabilitiesNetMinorityInterest",
            "annualStockholdersEquity",
            "annualCashAndCashEquivalents",
            "annualTotalDebt",
            "annualNetPPE",
            "annualCurrentAssets",
            "annualCurrentLiabilities",
            "annualInventory",
            "annualAccountsReceivable",
            "annualRetainedEarnings",
            "annualCommonStock",
            "annualTotalEquityGrossMinorityInterest",
            "annualGoodwill",
        };
        for (String type : required) {
            assertTrue(url.contains(type),
                "URL must include balance sheet type: " + type);
        }
    }

    @Test
    void buildUrl_containsAllRequiredCashFlowTypes() {
        String url = YfFinances.buildUrl("ANY.NZ");
        String[] required = {
            "annualCashFlowsfromusedinOperatingActivitiesDirect",
            "annualCapitalExpenditure",
            "annualFreeCashFlow",
            "annualCashFlowFromContinuingInvestingActivities",
            "annualCashFlowFromContinuingFinancingActivities",
            "annualCashDividendsPaid",
            "annualIssuanceOfDebt",
            "annualRepaymentOfDebt",
        };
        for (String type : required) {
            assertTrue(url.contains(type),
                "URL must include cash flow type: " + type);
        }
    }

    // ──────────────────────────────────────────────
    // getDataPoints — fetcher injection
    // ──────────────────────────────────────────────

    @Test
    void getDataPoints_fetcherReturningNull_doesNotThrowAndSkipsTicker() {
        YfFinances source = new YfFinances(url -> null);
        assertDoesNotThrow(source::getDataPoints);
    }

    @Test
    void getDataPoints_fetcherThrowingException_doesNotThrowAndContinues() {
        AtomicInteger calls = new AtomicInteger();
        String onePoint = validJson("FPH.NZ", TYPE, DATE, VALUE);
        YfFinances source = new YfFinances(url -> {
            if (calls.getAndIncrement() == 0) {
                throw new RuntimeException("simulated HTTP failure");
            }
            return onePoint;
        });
        Set<DataPoint> result = assertDoesNotThrow(source::getDataPoints);
        assertFalse(result.isEmpty(), "Non-failing tickers must still produce data points");
    }
}
