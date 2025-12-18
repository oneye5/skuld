package lazic.utils.ingest;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.time.ZoneId;

public class CsvLongParser {

	/**
	 * Saves all DataPoint entries to long-format CSV:
	 *
	 * timestamp_ms,ticker,feature,value
	 */
	public static boolean saveCsv(String path) {
		var data = IngestManager.INSTANCE.data;

		if (data.isEmpty()) {
			return false;
		}

		try (BufferedWriter bw = new BufferedWriter(new FileWriter(path))) {
			bw.write("timestamp,ticker,feature,value");
			bw.newLine();

			for (DataPoint dp : data) {
				long ts = (dp.getTimestamp() != null)
								? dp.getTimestamp()
								.atZone(ZoneId.of("UTC"))
								.toInstant()
								.toEpochMilli()
								: -1;

				String ticker = dp.getTicker() != null ? dp.getTicker() : "null";
				String feature = dp.getFeatureName() != null ? dp.getFeatureName() : "";
				String value = dp.getValue() != null ? dp.getValue().toString() : "";

				// clean of dangerous characters
				ticker = ticker.replace(",","-");
				ticker = ticker.replace("\n","-");

				feature = feature.replace(",","-");
				feature = feature.replace("\n","-");

				value = value.replace(",","-");
				value = value.replace("\n","-");

				bw.write(ts + "," + ticker + "," + feature + "," + value);
				bw.newLine();
			}

			return true;

		} catch (IOException e) {
			e.printStackTrace();
			return false;
		}
	}
}
