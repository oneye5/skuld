package lazic.sources;

import lazic.utils.ingest.DataPoint;
import org.junit.jupiter.api.Test;

import java.util.Set;
import java.util.function.Function;

import static org.junit.jupiter.api.Assertions.*;

class YfSectorTest {

    // ──────────────────────────────────────────────
    // Helper — minimal valid finance/search JSON
    // ──────────────────────────────────────────────

    private static String validJson(String ticker, String sector, String industry) {
        return "{"
            + "\"quotes\":[{"
            + "    \"symbol\":\"" + ticker + "\","
            + "    \"quoteType\":\"EQUITY\","
            + "    \"exchange\":\"NZE\","
            + "      \"sector\":\"" + sector + "\","
            + "      \"industry\":\"" + industry + "\""
            + "  }]"
            + "}";
    }

    // ──────────────────────────────────────────────
    // parseSector — null / empty / malformed
    // ──────────────────────────────────────────────

    @Test
    void parseSector_nullInput_returnsEmptySet() {
        assertTrue(YfSector.parseSector(null, "FPH.NZ").isEmpty(),
            "null JSON must not throw");
    }

    @Test
    void parseSector_emptyString_returnsEmptySet() {
        assertTrue(YfSector.parseSector("", "FPH.NZ").isEmpty());
    }

    @Test
    void parseSector_invalidJson_returnsEmptySetWithoutThrowing() {
        assertDoesNotThrow(() -> YfSector.parseSector("{bad json{{", "FPH.NZ"));
        assertTrue(YfSector.parseSector("{bad json{{", "FPH.NZ").isEmpty());
    }

    @Test
    void parseSector_missingQuotesKey_returnsEmptySet() {
        assertTrue(YfSector.parseSector("{}", "FPH.NZ").isEmpty());
    }

    @Test
    void parseSector_nullQuotesArray_returnsEmptySet() {
        String json = "{\"quotes\":null}";
        assertTrue(YfSector.parseSector(json, "FPH.NZ").isEmpty());
    }

    @Test
    void parseSector_emptyQuotesArray_returnsEmptySet() {
        String json = "{\"quotes\":[]}";
        assertTrue(YfSector.parseSector(json, "FPH.NZ").isEmpty());
    }

    @Test
    void parseSector_missingSectorField_returnsEmptySet() {
        String json = "{\"quotes\":[{\"symbol\":\"FPH.NZ\",\"industry\":\"Medical Instruments & Supplies\"}]}";
        assertTrue(YfSector.parseSector(json, "FPH.NZ").isEmpty(),
            "Missing sector field must not NPE");
    }

    // ──────────────────────────────────────────────
    // parseSector — happy path
    // ──────────────────────────────────────────────

    @Test
    void parseSector_healthcareSector_returnsGicsCode35() {
        Set<DataPoint> result = YfSector.parseSector(
            validJson("FPH.NZ", "Healthcare", "Medical Instruments & Supplies"), "FPH.NZ");

        assertFalse(result.isEmpty(), "Should return at least one DataPoint for known sector");
        DataPoint sectorDp = result.stream()
            .filter(dp -> "gics_sector".equals(dp.getFeatureName()))
            .findFirst()
            .orElseThrow(() -> new AssertionError("No gics_sector DataPoint found"));

        assertEquals("FPH.NZ", sectorDp.getTicker());
        assertEquals(35.0, sectorDp.getValue(), 0.001, "Healthcare must map to GICS code 35");
        assertNotNull(sectorDp.getTimestamp());
    }

    @Test
    void parseSector_allKnownSectors_mapToExpectedGicsCodes() {
        record Case(String yf, double gics) {}
        Case[] cases = {
            new Case("Energy",                  10.0),
            new Case("Basic Materials",         15.0),
            new Case("Industrials",             20.0),
            new Case("Consumer Cyclical",       25.0),
            new Case("Consumer Defensive",      30.0),
            new Case("Healthcare",              35.0),
            new Case("Financial Services",      40.0),
            new Case("Technology",              45.0),
            new Case("Communication Services",  50.0),
            new Case("Utilities",               55.0),
            new Case("Real Estate",             60.0),
        };
        for (Case c : cases) {
            Set<DataPoint> result = YfSector.parseSector(
                validJson("TEST.NZ", c.yf, "Any Industry"), "TEST.NZ");
            DataPoint dp = result.stream()
                .filter(p -> "gics_sector".equals(p.getFeatureName()))
                .findFirst()
                .orElseThrow(() -> new AssertionError("No gics_sector DataPoint for sector: " + c.yf));
            assertEquals(c.gics, dp.getValue(), 0.001,
                "Sector '" + c.yf + "' must map to GICS " + c.gics);
        }
    }

    @Test
    void parseSector_unknownSectorString_returnsEmptySet() {
        Set<DataPoint> result = YfSector.parseSector(
            validJson("FPH.NZ", "UnknownSectorXYZ", "Some Industry"), "FPH.NZ");
        assertTrue(result.stream().noneMatch(dp -> "gics_sector".equals(dp.getFeatureName())),
            "Unmapped sector string must be silently skipped");
    }

    @Test
    void parseSector_nullSectorField_returnsEmptySet() {
        String json = "{"
            + "\"quotes\":[{"
            + "  \"symbol\":\"FPH.NZ\","
            + "  \"sector\":null,"
            + "  \"industry\":\"Some Industry\""
            + "}]"
            + "}";
        assertDoesNotThrow(() -> YfSector.parseSector(json, "FPH.NZ"));
        assertTrue(YfSector.parseSector(json, "FPH.NZ").stream()
            .noneMatch(dp -> "gics_sector".equals(dp.getFeatureName())));
    }

    @Test
    void parseSector_emptySectorString_returnsEmptySet() {
        Set<DataPoint> result = YfSector.parseSector(validJson("FPH.NZ", "", "Some Industry"), "FPH.NZ");
        assertTrue(result.stream().noneMatch(dp -> "gics_sector".equals(dp.getFeatureName())));
    }

    @Test
    void parseSector_selectsExactTickerMatchFromQuotesArray() {
        String json = "{"
            + "\"quotes\":["
            + "  {\"symbol\":\"FPH\",\"sector\":\"Technology\",\"industry\":\"Other\"},"
            + "  {\"symbol\":\"FPH.NZ\",\"sector\":\"Healthcare\",\"industry\":\"Medical Instruments & Supplies\"}"
            + "]"
            + "}";

        Set<DataPoint> result = YfSector.parseSector(json, "FPH.NZ");
        DataPoint dp = result.stream()
            .filter(p -> "gics_sector".equals(p.getFeatureName()))
            .findFirst()
            .orElseThrow(() -> new AssertionError("No gics_sector DataPoint for exact ticker"));

        assertEquals(35.0, dp.getValue(), 0.001);
    }

    // ──────────────────────────────────────────────
    // buildUrl
    // ──────────────────────────────────────────────

    @Test
    void buildUrl_containsTickerInPath() {
        String url = YfSector.buildUrl("FPH.NZ");
        assertTrue(url.contains("FPH.NZ"), "URL must contain the ticker");
    }

    @Test
    void buildUrl_containsSearchQueryParameter() {
        String url = YfSector.buildUrl("FPH.NZ");
        assertTrue(url.contains("q=FPH.NZ"),
            "URL must query the ticker through Yahoo finance search");
    }

    // ──────────────────────────────────────────────
    // getDataPoints — fetcher injection
    // ──────────────────────────────────────────────

    @Test
    void getDataPoints_fetcherReturningNull_doesNotThrow() {
        YfSector source = new YfSector(url -> null);
        assertDoesNotThrow(source::getDataPoints);
    }
}
