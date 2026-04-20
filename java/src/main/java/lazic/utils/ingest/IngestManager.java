package lazic.utils.ingest;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

// singleton
public enum IngestManager {
	INSTANCE;
	public final Set<DataSourceBase> sources = new HashSet<>();
	public final List<DataPoint> data = Collections.synchronizedList(new ArrayList<>());

	public void fetchDataFromSources() {
		data.clear();
		sources.parallelStream().forEach(source -> {
			var dataPoints = source.getDataPoints();
			String sourceName = source.getSourceName();
			dataPoints.stream()
							.filter(dp -> dp.getValue() != null)
							.forEach(dp -> {
								dp.setSource(sourceName);
								data.add(dp);
							});
		});
	}

	public void printSubset(int count) {
		var asList = new ArrayList<>(data);
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

