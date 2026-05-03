package lazic.sources;

import lazic.utils.ingest.DataPoint;
import org.junit.jupiter.api.Test;

import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.Map;
import java.util.Set;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

class NzLaborStatsTest {

    private String loadFixture() throws IOException {
        try (InputStream is = getClass().getResourceAsStream("/lazic/sources/sdmx_labor_fixture.json")) {
            assertNotNull(is, "Fixture file not found on classpath");
            return new String(is.readAllBytes(), StandardCharsets.UTF_8);
        }
    }

    @Test
    void parseObservations_noConflictingFeatureNames() throws IOException {
        NzLaborStats source = new NzLaborStats();
        Set<DataPoint> points = source.parseObservations(loadFixture());

        Map<String, Long> countByKey = points.stream()
            .collect(Collectors.groupingBy(
                dp -> dp.getTimestamp() + "|" + dp.getFeatureName(),
                Collectors.counting()
            ));

        countByKey.forEach((key, count) ->
            assertEquals(1L, count,
                "Duplicate (timestamp, feature) found: " + key + " appeared " + count + " times")
        );
    }

    @Test
    void parseObservations_nationalAndRegionalHaveDistinctNames() throws IOException {
        NzLaborStats source = new NzLaborStats();
        Set<DataPoint> points = source.parseObservations(loadFixture());

        Set<String> featureNames = points.stream()
            .map(DataPoint::getFeatureName)
            .collect(Collectors.toSet());

        // Fixture has dim[3]=0 (NAT) and dim[3]=1 (REG); buildFeatureName() appends "_r<dim3>" suffix.
        // National series: _r0_NZL, regional (Auckland) series: _r1_NZL_AUK.
        assertTrue(featureNames.stream().anyMatch(n -> n.contains("_r0")),
            "Expected a feature name with national suffix '_r0', got: " + featureNames);
        assertTrue(featureNames.stream().anyMatch(n -> n.contains("_r1")),
            "Expected a feature name with regional suffix '_r1', got: " + featureNames);
        assertEquals(2, featureNames.size(),
            "National and regional observations must produce 2 distinct feature names, got: " + featureNames);
    }

    @Test
    void parseObservations_returnsNonEmptySet() throws IOException {
        // Fixture year is 2005; ReleaseFilter.isKnowableNow checks timestamp <= now.
        // With a 12-month lag, the release timestamp is 2006-01-01 — safely in the past.
        NzLaborStats source = new NzLaborStats();
        Set<DataPoint> points = source.parseObservations(loadFixture());
        assertFalse(points.isEmpty(), "Should parse at least one observation from fixture");
    }
}
