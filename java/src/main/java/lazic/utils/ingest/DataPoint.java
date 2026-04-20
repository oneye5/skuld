package lazic.utils.ingest;
import java.io.Serializable;
import java.time.LocalDateTime;

public class DataPoint implements Serializable {

	private LocalDateTime timestamp;
	private String ticker;          // optional, may be null for macro data
	private String featureName;     // e.g. "close_price", "inflation", "EPS"
	private Double value;           // exactly one feature
	private String source;          // data source identifier, set by IngestManager

	public DataPoint() {}

	public DataPoint(LocalDateTime timestamp,
									 String ticker,
									 String featureName,
									 Double value) {
		this.timestamp = timestamp;
		this.ticker = ticker;
		this.featureName = featureName;
		this.value = value;
	}

	public LocalDateTime getTimestamp() { return timestamp; }
	public void setTimestamp(LocalDateTime timestamp) { this.timestamp = timestamp; }

	public String getTicker() { return ticker; }
	public void setTicker(String ticker) { this.ticker = ticker; }

	public String getFeatureName() { return featureName; }
	public void setFeatureName(String featureName) { this.featureName = featureName; }

	public Double getValue() { return value; }
	public void setValue(double value) { this.value = value; }

	public String getSource() { return source; }
	public void setSource(String source) { this.source = source; }

	@Override
	public String toString() {
		return String.format(
						"DataPoint(timestamp=%s, ticker=%s, feature=%s, value=%s, source=%s)",
						timestamp, ticker, featureName, value, source
		);
	}
}
