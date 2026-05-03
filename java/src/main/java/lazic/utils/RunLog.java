package lazic.utils;

import java.io.IOException;
import java.io.OutputStream;
import java.io.PrintStream;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

/**
 * Captures all runtime stderr output and uncaught exceptions to a log file
 * alongside the data output, while continuing to mirror them to the console.
 *
 * <p>Install once at process start via {@link #install(Path)}. After that:
 * <ul>
 *   <li>Every write to {@link System#err} (including {@code e.printStackTrace()})
 *       is tee'd into the log file.</li>
 *   <li>Any uncaught exception on any thread is written to the log with a
 *       full stack trace and a header marking it as fatal.</li>
 *   <li>A JVM shutdown hook flushes and closes the log cleanly.</li>
 * </ul>
 *
 * <p>The log is opened in truncate mode so each run starts fresh, matching
 * the overwrite semantics of the data CSV.
 */
public final class RunLog {

	private static final DateTimeFormatter TS = DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss");

	private static PrintStream originalErr;
	private static PrintStream fileStream;
	private static boolean installed = false;

	private RunLog() {}

	/**
	 * Wire stderr tee, uncaught-exception handler, and shutdown hook.
	 * Idempotent — subsequent calls are no-ops.
	 */
	public static synchronized void install(Path logFile) {
		if (installed) return;

		try {
			Files.createDirectories(logFile.getParent());
			OutputStream raw = Files.newOutputStream(
				logFile,
				StandardOpenOption.CREATE,
				StandardOpenOption.TRUNCATE_EXISTING,
				StandardOpenOption.WRITE);
			fileStream = new PrintStream(raw, true);
		} catch (IOException e) {
			// Cannot open the log file — fall back to console-only and surface why.
			System.err.println("RunLog: failed to open " + logFile + ": " + e.getMessage());
			return;
		}

		originalErr = System.err;
		System.setErr(new PrintStream(new TeeOutputStream(originalErr, fileStream), true));

		Thread.setDefaultUncaughtExceptionHandler((thread, throwable) -> {
			System.err.println();
			System.err.println("[" + LocalDateTime.now().format(TS) + "] UNCAUGHT EXCEPTION on thread '" + thread.getName() + "':");
			throwable.printStackTrace(System.err);
		});

		Runtime.getRuntime().addShutdownHook(new Thread(() -> {
			try {
				if (fileStream != null) {
					fileStream.println();
					fileStream.println("[" + LocalDateTime.now().format(TS) + "] --- run end ---");
					fileStream.flush();
					fileStream.close();
				}
			} catch (Exception ignored) {
				// Shutdown — nothing to do.
			}
		}, "RunLog-shutdown"));

		fileStream.println("[" + LocalDateTime.now().format(TS) + "] --- run start ---");
		installed = true;
	}

	/** Tees writes to two underlying streams. */
	private static final class TeeOutputStream extends OutputStream {
		private final OutputStream a;
		private final OutputStream b;

		TeeOutputStream(OutputStream a, OutputStream b) {
			this.a = a;
			this.b = b;
		}

		@Override
		public void write(int byteVal) throws IOException {
			a.write(byteVal);
			b.write(byteVal);
		}

		@Override
		public void write(byte[] buf, int off, int len) throws IOException {
			a.write(buf, off, len);
			b.write(buf, off, len);
		}

		@Override
		public void flush() throws IOException {
			a.flush();
			b.flush();
		}

		@Override
		public void close() throws IOException {
			try { a.flush(); } catch (IOException ignored) {}
			try { b.flush(); } catch (IOException ignored) {}
			// Do not close the originals — `a` is the JVM's stderr.
		}
	}
}
