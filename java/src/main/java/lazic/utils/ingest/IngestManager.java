package lazic.utils.ingest;

import java.util.Collections;
import java.util.HashSet;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

// singleton
public enum IngestManager {
	INSTANCE;
	public final Set<DataSourceBase> sources = new HashSet<>();
	// Use ConcurrentHashMap-backed Set for thread-safe parallel writes
	public final Set<DataPoint> data = ConcurrentHashMap.newKeySet();

	public void fetchDataFromSources() {
		data.clear();
		sources.parallelStream().forEach(source -> {
			var dataPoints = source.getDataPoints();
			dataPoints = dataPoints.stream()
							.filter(dp->dp.getValue() != null)
							.collect(Collectors.toSet());

			this.data.addAll(dataPoints);
		});
	}

	public void printSubset(int count) {
		// Create a snapshot to avoid concurrent modification
		var asList = new java.util.ArrayList<>(new HashSet<>(data));
		Collections.shuffle(asList);

		asList.subList(0, Math.min(asList.size(), count))
						.forEach(dp-> System.out.println(dp.toString()));
	}
	}

	/**
	 * datapoint contents:
	 *
	 * public DataPoint(LocalDateTime timestamp,
	 * 									 String ticker,
	 * 									 String featureName,
	 * 									 Double value)
	 */
}
