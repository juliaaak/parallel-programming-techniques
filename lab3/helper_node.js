/**
 * Lab 3 - Task 2, Node.js helper process
 * Acts as a TCP server: receives a float (8 bytes, little-endian double),
 * logs it, and echoes it back. Demonstrates cross-language IPC via socket.
 */

const net = require('net');

const PORT = 54322;  // different port from Python-to-Python socket test
const LOG_FIRST_N = 3;
let count = 0;

const server = net.createServer((socket) => {
    let buffer = Buffer.alloc(0);

    socket.on('data', (chunk) => {
        buffer = Buffer.concat([buffer, chunk]);

        // process all complete 8-byte messages in the buffer
        while (buffer.length >= 8) {
            const value = buffer.readDoubleLE(0);
            buffer = buffer.slice(8);

            count++;
            if (count <= LOG_FIRST_N) {
                process.stdout.write(`  [Node.js helper] received ${value.toFixed(6)}\n`);
            } else if (count === LOG_FIRST_N + 1) {
                process.stdout.write(`  [Node.js helper] ... (logging first ${LOG_FIRST_N} only)\n`);
            }

            // echo the value back as 8-byte little-endian double
            const response = Buffer.alloc(8);
            response.writeDoubleLE(value, 0);
            socket.write(response);
        }
    });

    socket.on('end', () => {
        server.close();
    });
});

server.listen(PORT, '127.0.0.1', () => {
    // signal to parent Python process that server is ready
    process.stdout.write('READY\n');
});
