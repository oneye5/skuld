package lazic.sources.examples;

import com.google.gson.Gson;
import lazic.utils.ingest.DataPoint;
import lazic.utils.ingest.DataSourceBase;
import lazic.utils.ingest.WebHtmlGetter;

import java.io.FileWriter;
import java.io.IOException;
import java.time.LocalDateTime;
import java.util.Set;

public class SourceTemplate extends DataSourceBase {
	private final String URL = "";

	/**
	 * Returns a set of DataPoint's. Ticker is null if the datapoint does not pertain to a particular ticker, such as macroeconomic data for example
	 * There are multiple DataPoint's in a time-series feature, and there may be multiple features returned overall.
	 */
	@Override
	public Set<DataPoint> getDataPoints() {
		DataPoint example = new DataPoint(LocalDateTime.now(), "Ticker", "Feature name", Double.valueOf(1));

		String[] tickers = lazic.sources.config.Tickers.TICKERS; //"ANZ.NZ", "AFCA.NZ", etc
		Gson gson = new Gson();
		String rawData = WebHtmlGetter.get(URL);
		System.out.println(rawData);

		try (FileWriter writer = new FileWriter("sample_data.txt")) {
			writer.write(rawData);
		} catch (IOException e) {
			throw new RuntimeException(e);
		}
		return Set.of(example);
	}
}
