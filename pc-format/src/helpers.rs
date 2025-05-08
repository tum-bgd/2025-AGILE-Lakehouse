/// convert a parallel iterator into an iterator
pub fn into_iter<T: Send + 'static>(
    iter: impl rayon::iter::ParallelIterator<Item = T> + 'static,
) -> impl Iterator<Item = T> {
    let num_cpus = std::thread::available_parallelism().unwrap().get();

    let (send, recv) = std::sync::mpsc::sync_channel(num_cpus);

    std::thread::spawn(move || {
        iter.for_each(|item| {
            let _ = send.send(item);
        })
    });

    recv.into_iter()
}
