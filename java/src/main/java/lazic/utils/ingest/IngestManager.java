package lazic.utils.ingest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

// singleton
public enum IngestManager {
	INSTANCE;
	public final Set<DataSourceBase> sources = new HashSet<>();
	public final List<DataPoint> data = Collections.synchronizedList(new ArrayList<>());

	/**
	 * Fetches data from all registered sources in parallel.
	 * Prints a per-source summary on completion (point count, or failure reason).
	 *
	 * @return number of sources that failed (threw an exception)
	 */
	public int fetchDataFromSources() {
		data.clear();
		Map<String, Integer> pointCounts = new ConcurrentHashMap<>();
		Map<String, String> failures = new ConcurrentHashMap<>();
		AtomicInteger failureCount = new AtomicInteger(0);

		sources.parallelStream().forEach(source -> {
			String sourceName = source.getSourceName();
			try {
				var dataPoints = source.getDataPoints();
				int added = 0;
				for (DataPoint dp : dataPoints) {
					if (dp.getValue() != null) {
						dp.setSource(sourceName);
						data.add(dp);
						added++;
					}
				}
				pointCounts.put(sourceName, added);
			} catch (Exception e) {
				failures.put(sourceName, e.getClass().getSimpleName() + ": " + e.getMessage());
				failureCount.incrementAndGet();
			}
		});

		// Print run summary to stderr so it is always visible regardless of
		// stdout buffering, and separable from data output.
		System.err.println("=== IngestManager run summary ===");
		for (DataSourceBase src : sources) {
			String name = src.getSourceName();
			if (failures.containsKey(name)) {
				System.err.println("  [FAIL] " + name + " — " + failures.get(name));
			} else {
				System.err.println("  [OK]   " + name + " — " + pointCounts.getOrDefault(name, 0) + " points");
			}
		}
		System.err.println("  Total: " + data.size() + " points, " + failureCount.get() + " source(s) failed");
		System.err.println("=================================");

		return failureCount.get();
	}

	public void printSubset(int count) {
		var asList = new ArrayList<>(data);
		Collections.shuffle(asList);

		asList.subList(0, Math.min(asList.size(), count))
						.forEach(dp-> System.out.println(dp.toString()));
	}
}
