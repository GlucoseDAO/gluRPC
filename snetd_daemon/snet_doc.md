# Configuring the Daemon

> **Note:** This section is only for **self-hosted daemon deployment**. Skip this if using HaaS.

To run the daemon, create and edit the configuration file named `snetd.config.json`. This file tells the daemon how to communicate with your AI service, blockchain, and payment storage.

Edit the configuration file:

```sh
$EDITOR snetd.config.json
```

## Example Configuration

Below are complete configuration examples for **Mainnet** and **Testnet (Sepolia)**. Replace all placeholders (`<...>`) accordingly.

### Testnet

```json
{
  "blockchain_enabled": true,
  "blockchain_network_selected": "sepolia",

  "daemon_endpoint": "0.0.0.0:<DAEMON_PORT>",
  "daemon_group_name": "<DAEMON_GROUP>",

  "organization_id": "<ORGANIZATION_ID>",
  "service_id": "<SERVICE_ID>",
  "service_endpoint": "http://<SERVICE_HOST>:<SERVICE_PORT>",

  "ssl_cert": "<PATH_TO_DOMAIN_CERTS>/fullchain.pem",
  "ssl_key": "<PATH_TO_DOMAIN_CERTS>/privkey.pem",

  "metering_enabled": true,
  "metering_endpoint": "https://marketplace-mt-v2.singularitynet.io",
  "private_key_for_metering": "<METERING_KEY>",

  "private_key_for_free_calls": "<FREE_CALL_KEY>",

  "ethereum_json_rpc_http_endpoint": "https://eth-sepolia.g.alchemy.com/v2/<YOUR_API_KEY>",
  "ethereum_json_rpc_ws_endpoint": "wss://eth-sepolia.g.alchemy.com/v2/<YOUR_API_KEY>",

  "payment_channel_storage_server": {
    "client_port": 2379,
    "cluster": "storage-1=http://127.0.0.1:2380",
    "data_dir": "data.etcd",
    "enabled": true,
    "host": "127.0.0.1",
    "id": "storage-1",
    "log_level": "info",
    "peer_port": 2380,
    "scheme": "http",
    "startup_timeout": "1m",
    "token": "your-unique-token"
  },

  "log": {"level": "debug", "output": {"type": "stdout"}}
}
```

### Mainnet

```json
{
  "blockchain_enabled": true,
  "blockchain_network_selected": "main",

  "daemon_endpoint": "0.0.0.0:<DAEMON_PORT>",
  "daemon_group_name": "<DAEMON_GROUP>",

  "organization_id": "<ORGANIZATION_ID>",
  "service_id": "<SERVICE_ID>",
  "service_endpoint": "http://<SERVICE_HOST>:<SERVICE_PORT>",

  "ssl_cert": "<PATH_TO_DOMAIN_CERTS>/fullchain.pem",
  "ssl_key": "<PATH_TO_DOMAIN_CERTS>/privkey.pem",

  "metering_enabled": true,
  "metering_endpoint": "https://marketplace-mt-v2.singularitynet.io",
  "private_key_for_metering": "<METERING_KEY>",

  "private_key_for_free_calls": "<FREE_CALL_KEY>",

  "ethereum_json_rpc_http_endpoint": "https://eth-mainnet.g.alchemy.com/v2/<YOUR_API_KEY>",
  "ethereum_json_rpc_ws_endpoint": "wss://eth-mainnet.g.alchemy.com/v2/<YOUR_API_KEY>",

  "payment_channel_storage_server": {
    "client_port": 2379,
    "cluster": "storage-1=http://127.0.0.1:2380",
    "data_dir": "data.etcd",
    "enabled": true,
    "host": "127.0.0.1",
    "id": "storage-1",
    "log_level": "info",
    "peer_port": 2380,
    "scheme": "http",
    "startup_timeout": "1m",
    "token": "your-unique-token"
  },

  "log": {"level": "debug", "output": {"type": "stdout"}}
}
```

> **DANGER**
> 
> For each reference to the embedded ETCD configuration in the daemon, do not delete the directory specified by `data_dir`. Deleting this folder will remove access to payment channel storage and prevent token withdrawals.

---

## Placeholders to Replace

| Placeholder               | Explanation                                                                   |
|---------------------------|-------------------------------------------------------------------------------|
| `<DAEMON_PORT>`           | Port number where the daemon will run (e.g., 7000)                            |
| `<DAEMON_GROUP>`          | Group name for your daemon (default_group)                                     |
| `<ORGANIZATION_ID>`       | Your organization's ID (after publishing organization)                         |
| `<SERVICE_ID>`            | Your service's ID (after publishing service)                                   |
| `<SERVICE_HOST>`          | Address (IP or hostname) of your running AI service                            |
| `<SERVICE_PORT>`          | Port number on which your AI service is listening                              |
| `<PATH_TO_DOMAIN_CERTS>`  | Directory containing your domain certificates (fullchain.pem and privkey.pem)  |
| `<METERING_KEY>`          | Previously generated private key for metering                                  |
| `<FREE_CALL_KEY>`         | Previously generated private key for free calls                                |
| `<YOUR_API_KEY>`          | Alchemy API key for blockchain communication                                   |

---

## Configuration Field Explanations

| Field                               | Explanation                                                           |
|-------------------------------------|-----------------------------------------------------------------------|
| `blockchain_enabled`                | Enables blockchain connectivity (always true)                         |
| `blockchain_network_selected`       | Blockchain network (main for Mainnet, sepolia for Testnet)            |
| `daemon_endpoint`                   | Address and port where daemon listens for incoming connections        |
| `daemon_group_name`                 | Name of payment group (defined earlier)                               |
| `organization_id`                   | ID referencing your published organization                            |
| `service_id`                        | ID referencing your published AI service                              |
| `service_endpoint`                  | Internal endpoint for your AI service                                 |
| `ssl_cert` and `ssl_key`            | SSL certificate paths for secure connections to daemon                |
| `metering_enabled`                  | Activates request metering functionality (true)                       |
| `metering_endpoint`                 | Endpoint for metering service (no changes required)                   |
| `private_key_for_metering`          | Ethereum private key for metering functionality                       |
| `private_key_for_free_calls`        | Ethereum private key for free calls functionality                     |
| `ethereum_json_rpc_http_endpoint`   | HTTP RPC endpoint for blockchain communication                        |
| `ethereum_json_rpc_ws_endpoint`     | WebSocket RPC endpoint for blockchain communication                   |
| `payment_channel_storage_server`    | Embedded ETCD setup (no modification required if using embedded ETCD) |
| `log`                               | Daemon logging settings                                               |

---

## Switching Between Testnet and Mainnet

When moving from Sepolia testnet to Ethereum mainnet (or vice versa), you need to update specific parameters in your daemon configuration.

### Parameters to Change

| Parameter                           | Testnet (Sepolia)                          | Mainnet                                    |
|-------------------------------------|--------------------------------------------|--------------------------------------------|
| `blockchain_network_selected`       | `"sepolia"`                                | `"main"`                                   |
| `ethereum_json_rpc_http_endpoint`   | `https://eth-sepolia.g.alchemy.com/v2/<KEY>` | `https://eth-mainnet.g.alchemy.com/v2/<KEY>` |
| `ethereum_json_rpc_ws_endpoint`     | `wss://eth-sepolia.g.alchemy.com/v2/<KEY>`   | `wss://eth-mainnet.g.alchemy.com/v2/<KEY>`   |

### Additional Considerations

- **Organization and Service IDs** must be re-registered on the target network (testnet registrations do not transfer to mainnet)
- **Payment Address** should be verified for the target network
- **Metering Endpoint** remains the same for both networks (`https://marketplace-mt-v2.singularitynet.io`)
- **ETCD Configuration** does not change between networks

### Switching Checklist

1. Update `blockchain_network_selected` in daemon config
2. Update both Alchemy RPC endpoints (HTTP and WebSocket)
3. Re-register your organization on the target network
4. Re-publish your service on the target network
5. Update `organization_id` and `service_id` in daemon config with new values
6. Restart the daemon

---

## Daemon Setup Summary

> **Note:** This summary is for **self-hosted daemon deployment** only.

At this point, the daemon cannot be started because the organization and service are not yet created. These steps will be covered next. You have completed:

- Domain and SSL certificate configuration
- Metering and free calls private key generation
- Daemon installation and preparation
- Configuration file setup

Proceed to the next step to create your Organization and Service.

If you prefer not to manage infrastructure, consider using Hosting-as-a-Service (HaaS) available through the Publisher Portal.
