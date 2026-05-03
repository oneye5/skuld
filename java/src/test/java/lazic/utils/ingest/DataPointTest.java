package lazic.utils.ingest;

import org.junit.jupiter.api.Test;
import java.time.LocalDateTime;
import java.util.HashSet;
import java.util.Set;

import static org.junit.jupiter.api.Assertions.*;

class DataPointTest {

    private static final LocalDateTime TS = LocalDateTime.of(2024, 1, 1, 0, 0);

    @Test
    void equalPoints_areEqual() {
        DataPoint a = new DataPoint(TS, "NZX:AIR", "close_price", 2.50);
        DataPoint b = new DataPoint(TS, "NZX:AIR", "close_price", 2.50);
        assertEquals(a, b);
    }

    @Test
    void equalPoints_haveSameHashCode() {
        DataPoint a = new DataPoint(TS, "NZX:AIR", "close_price", 2.50);
        DataPoint b = new DataPoint(TS, "NZX:AIR", "close_price", 2.50);
        assertEquals(a.hashCode(), b.hashCode());
    }

    @Test
    void differentValue_notEqual() {
        DataPoint a = new DataPoint(TS, null, "unemployment_rate", 11.3);
        DataPoint b = new DataPoint(TS, null, "unemployment_rate", 18400.0);
        assertNotEquals(a, b);
    }

    @Test
    void hashSet_deduplicatesExactDuplicates() {
        DataPoint a = new DataPoint(TS, null, "inflation", 3.5);
        DataPoint b = new DataPoint(TS, null, "inflation", 3.5);
        Set<DataPoint> set = new HashSet<>();
        set.add(a);
        set.add(b);
        assertEquals(1, set.size(), "HashSet must collapse identical DataPoints to one entry");
    }

    @Test
    void hashSet_keepsDistinctValues() {
        DataPoint rate = new DataPoint(TS, null, "unemployment", 11.3);
        DataPoint count = new DataPoint(TS, null, "unemployment", 18400.0);
        Set<DataPoint> set = new HashSet<>();
        set.add(rate);
        set.add(count);
        assertEquals(2, set.size(), "DataPoints with different values must remain distinct in HashSet");
    }

    @Test
    void nullTicker_handledCorrectly() {
        DataPoint a = new DataPoint(TS, null, "gdp", 200000.0);
        DataPoint b = new DataPoint(TS, null, "gdp", 200000.0);
        assertEquals(a, b);
        assertEquals(a.hashCode(), b.hashCode());
    }
}
